#include "prism.h"
#include "dsp.h"
#include "rmvpe_onnx.h"
#include "hubert_encoder.h"
#include "yin.h"

#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <future>
#include <thread>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <cstdlib>

void ml_thread_stop(AppState* s);
struct PrismModel;  // forward decl — kept for any callers that still include ml_thread.h

static constexpr int kRate48        = 48000;
static constexpr int kRate16        = 16000;
// 150ms hop: a 100ms hop left zero compute margin on a 6-core CPU (measured
// 96-105ms/chunk → constant catch-up dropouts). Per-hop cost grows slower
// than the budget (encoder/tracker windows are fixed), so 150ms buys ~35%
// headroom for +50ms latency.
static constexpr int kHopSamples48 = 7200;      // 150ms @ 48kHz
static constexpr int kHopSamples16 = 2400;      // 150ms @ 16kHz
static constexpr int kWinSamples16 = 5120;      // 320ms @ 16kHz  (HuBERT context)
static constexpr int kFramesPerHop = kHopSamples16 / 160;  // 10ms RMVPE frames

// Instead of synthesising every sample twice (50% OLA) we synthesise a short
// chunk and crossfade the seam against the previous call's tail. The decoder's
// conv edges corrupt roughly the first and last frame of every chunk
// (measured: frame 0 corr 0.15 vs a continuous render, last frame 0.24,
// interior ~0.9), so one guard frame per side is synthesised and discarded.
static constexpr int kGuardFrames = 1;
static constexpr int kXfadeFrames = 2;                              // 20ms
static constexpr int kSynFrames   = kGuardFrames + kFramesPerHop +
                                    kXfadeFrames + kGuardFrames;    // 19 → 190ms
static constexpr int kGuard48     = kGuardFrames * 480;             // @48kHz
static constexpr int kXfade48     = kXfadeFrames * 480;
static constexpr int kRmvpeWin16  = 4800;   // 300ms → 31 frames, pads to 32

static std::string models_dir() {
    const char* home = getenv("HOME");
    if (!home) home = "";
    return std::string(home) + "/.cache/prism/models";
}

// ── Auto octave — trained-register coverage ──────────────────────────────────
// Same coarse-bin mapping as RVC's f0_to_coarse: mel = 1127·ln(1 + f0/700),
// bins 1..255 spanning 50..1100 Hz. Coverage = fraction of recent voiced
// frames that land in the model's trained bins after the candidate shift.

static float mask_coverage(const uint8_t* mask, const float* hist, int n,
                           int shift) {
    const float mel_min = 1127.f * std::log(1.f + 50.f   / 700.f);
    const float mel_max = 1127.f * std::log(1.f + 1100.f / 700.f);
    float mult = std::pow(2.f, shift / 12.f);
    int voiced = 0, in_mask = 0;
    for (int i = 0; i < n; i++) {
        float v = hist[i];
        if (v <= 0.f) continue;
        voiced++;
        float mel = 1127.f * std::log(1.f + v * mult / 700.f);
        float b   = (mel - mel_min) * 254.f / (mel_max - mel_min) + 1.f;
        int   bin = (int)std::lround(std::fmin(255.f, std::fmax(1.f, b)));
        if (mask[bin]) in_mask++;
    }
    return voiced ? (float)in_mask / (float)voiced : 0.f;
}

// ── Note segmenter — f0 frames → NoteEvents ──────────────────────────────────
// Runs on the 10ms RMVPE frame clock. A note opens after 3 consecutive frames
// agree on a pitch class; it closes after 5 unvoiced frames or when the pitch
// settles ≥ 0.75 semitones away.

struct NoteSegmenter {
    int    cur_note  = -1;
    float  cur_on    = 0.f;
    int    cand_note = -1;
    int    cand_run  = 0;
    int    off_run   = 0;

    void close(AppState* s, float t_off) {
        if (cur_note < 0) return;
        std::lock_guard<std::mutex> lk(s->notes_mu);
        s->notes.push_back({cur_on, t_off, cur_note});
        cur_note = -1;
    }

    void feed(AppState* s, float f0, float t) {
        if (f0 > 50.f) {
            off_run = 0;
            float nf = 69.f + 12.f * std::log2(f0 / 440.f);
            if (cur_note >= 0 && std::fabs(nf - (float)cur_note) < 0.75f) {
                cand_run = 0;
                return;   // still on the same note
            }
            int n = (int)std::lround(nf);
            if (n == cand_note) cand_run++;
            else { cand_note = n; cand_run = 1; }
            if (cand_run >= 3) {
                float onset = t - 0.03f;
                close(s, onset);
                cur_note = cand_note;
                cur_on   = onset;
                cand_run = 0;
            }
        } else {
            cand_note = -1;
            cand_run  = 0;
            if (cur_note >= 0 && ++off_run >= 5)
                close(s, t - 0.05f);
        }
    }
};

static void ml_thread_main(AppState* s) {
    s->vits_ring_ready.store(false);
    s->crossfade = 0.f;
    s->vits_fade  = 0.f;
    s->ml_frames.store(0);
    s->live_note.store(-1);
    s->live_f0.store(0.f);
    s->auto_octave.store(0);
    {
        std::lock_guard<std::mutex> lk(s->notes_mu);
        s->notes.clear();
    }

    s->model_state.store(2);  // loading ONNX sessions
    snprintf(s->status_msg, sizeof(s->status_msg), "Loading model\u2026");

    // hardware_concurrency counts SMT siblings; ONNX conv kernels scale with
    // physical cores and degrade past them (measured: VITS 170ms @ 4 threads
    // vs 140ms @ 6 on a 6C/12T part, worse at 8+).
    int n_threads = (int)std::thread::hardware_concurrency();
    if (n_threads < 2) n_threads = 2;
    if (n_threads > 4) n_threads /= 2;   // assume SMT

    // Load HuBERT (ContentVec) for phone features — the standard RVC content
    // encoder. Much better phonetic preservation than wav2vec2.
    HubertEncoder hubert;
    try {
        hubert.load(models_dir() + "/hubert.onnx", std::max(2, n_threads / 2));
    } catch (...) {
        snprintf(s->status_msg, sizeof(s->status_msg),
                 "hubert.onnx missing — copy from pop-maker-studio/build/models/");
        s->model_state.store(4);
        s->ml_running.store(false);
        return;
    }

    // RMVPE neural pitch tracker — YIN fallback if the model is absent
    RmvpeOnnx rmvpe;
    rmvpe.load(models_dir() + "/rmvpe.onnx", std::max(2, n_threads / 2));
    s->rmvpe_active.store(rmvpe.loaded());

    // Load VITS ONNX

    Ort::Env            venv{ORT_LOGGING_LEVEL_WARNING, "vits"};
    Ort::SessionOptions vopts;
    vopts.SetIntraOpNumThreads(n_threads);
    vopts.SetInterOpNumThreads(1);
    vopts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    vopts.AddConfigEntry("session.intra_op.allow_spinning", "0");
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::unique_ptr<Ort::Session> vits;
    try {
        vits = std::make_unique<Ort::Session>(venv, s->vits_onnx_path.c_str(), vopts);
    } catch (const std::exception& e) {
        snprintf(s->status_msg, sizeof(s->status_msg), "VITS load error: %.80s", e.what());
        s->model_state.store(4);
        s->ml_running.store(false);
        return;
    }

    s->model_state.store(3);  // session loaded → ready
    snprintf(s->status_msg, sizeof(s->status_msg), "Model loaded");

    int model_sr = s->output_sr.load();

    // Trained-register mask, eroded by ±4 bins (~3 st): pitch hugging the rim
    // of the trained register sounds strained even though it's "covered".
    uint8_t core_mask[256] = {};
    bool have_mask = s->have_register_mask.load();
    if (have_mask) {
        for (int b = 0; b < 256; b++) {
            bool ok = true;
            for (int k = -4; k <= 4 && ok; k++) {
                int j = b + k;
                ok = (j >= 0 && j < 256) ? (s->register_mask[j] != 0) : false;
            }
            core_mask[b] = ok ? 1 : 0;
        }
    }

    std::vector<float> win16(kWinSamples16, 0.f);
    std::vector<float> hop48(kHopSamples48);
    std::vector<float> hop16(kHopSamples16 + 16);
    std::vector<float> out48((kSynFrames + 2) * 480 * 2);
    std::vector<float> xf_tail(kXfade48, 0.f);  // previous call's last 20ms
    float ms_ema = 0.f;

    // Rolling voiced-f0 history for auto-octave (~3s of 10ms frames)
    static constexpr int kHistLen = 300;
    std::vector<float> f0_hist(kHistLen, 0.f);
    int hist_pos = 0;
    int cur_shift = 0;
    int pending_shift = 0;
    int hops_since_switch = 1000;

    // Streaming sine phase (cycles, mod 1) — fed to the graph's "phase" input
    // when present so each chunk continues the NSF oscillator where the
    // Probe for optional inputs: phase (v2 streaming) and rnd (v3 noise sampling)
    bool has_phase_in = false;
    bool has_rnd_in   = false;
    int  rnd_channels = 192;  // cfg.inter_channels default
    {
        Ort::AllocatorWithDefaultOptions alloc;
        size_t n_in = vits->GetInputCount();
        for (size_t i = 0; i < n_in; i++) {
            std::string nm = vits->GetInputNameAllocated(i, alloc).get();
            if (nm == "phase") has_phase_in = true;
            if (nm == "rnd") {
                has_rnd_in = true;
                auto shape = vits->GetInputTypeInfo(i)
                                 .GetTensorTypeAndShapeInfo().GetShape();
                if (shape.size() == 3 && shape[1] > 0)
                    rnd_channels = (int)shape[1];
            }
        }
    }
    float sine_phase = 0.f;

    // Seeded RNG for reproducible noise across runs (standard RVC seed)
    std::mt19937 rng(0x9e3779b9u);
    std::normal_distribution<float> gauss(0.f, 1.f);

    // Silence gate: RVC hallucinates breathy garbage on near-silent input, so
    // below the RMS threshold we emit actual silence and skip inference
    // entirely. Opens on one loud hop, closes after 2 quiet ones (200ms).
    // Threshold is UI-adjustable (mic levels vary wildly); close = open/2.
    int quiet_hops = 0;
    bool gated = true;   // start closed — the window is zeros anyway
    bool was_gated = true;

    NoteSegmenter seg;
    float smoothed_f0 = 0.f;

    while (s->ml_running.load()) {
        if (s->input_ring.available() < (size_t)kHopSamples48) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();

        s->input_ring.pull(hop48.data(), kHopSamples48);

        // Resample 48k → 16k with 3-tap boxcar anti-alias (3:1 ratio)
        size_t n16 = 0;
        for (size_t i = 0; i + 3 <= (size_t)kHopSamples48; i += 3)
            hop16[n16++] = (hop48[i] + hop48[i+1] + hop48[i+2]) * 0.33333333f;

        // Slide 400ms window
        std::memmove(win16.data(), win16.data() + n16,
                     (kWinSamples16 - n16) * sizeof(float));
        std::memcpy(win16.data() + (kWinSamples16 - n16), hop16.data(),
                    n16 * sizeof(float));

        // Normalise window to 0.95 peak for consistent HuBERT features
        {
            float peak = 0.f;
            for (int i = 0; i < kWinSamples16; ++i)
                peak = std::max(peak, std::abs(win16[i]));
            if (peak > 0.001f) {
                float gain = 0.95f / peak;
                for (int i = 0; i < kWinSamples16; ++i)
                    win16[i] *= gain;
            }
        }

        // Silence gate — emit real silence instead of letting VITS hallucinate
        // on a near-silent window, and skip all inference while closed.
        {
            double acc = 0.0;
            for (size_t i = 0; i < n16; ++i) acc += (double)hop16[i] * hop16[i];
            float rms = (float)std::sqrt(acc / (double)(n16 ? n16 : 1));
            s->input_rms.store(rms);
            float thr = s->gate_threshold.load();
            if (gated) {
                if (rms > thr) { gated = false; quiet_hops = 0; }
            } else if (rms < thr * 0.5f) {
                if (++quiet_hops >= 2) gated = true;
            } else {
                quiet_hops = 0;
            }
            s->gate_open.store(!gated);
            if (!gated && was_gated) smoothed_f0 = 0.f;
            was_gated = gated;
        }
        if (gated) {
            // Fade the pending crossfade tail into silence, then zeros.
            std::fill(out48.begin(), out48.begin() + kHopSamples48, 0.f);
            for (int i = 0; i < kXfade48; ++i)
                out48[(size_t)i] = xf_tail[(size_t)i] * (1.f - (float)i / kXfade48);
            std::memset(xf_tail.data(), 0, kXfade48 * sizeof(float));
            s->output_ring.push(out48.data(), kHopSamples48);
            sine_phase = 0.f;

            // Silence is the safe moment to change register — inaudible.
            if (cur_shift != pending_shift) {
                cur_shift = pending_shift;
                hops_since_switch = 0;
                s->auto_octave.store(cur_shift);
            }

            // Keep the transcription clock moving so open notes close.
            uint64_t fb = s->ml_frames.load();
            for (int i = 0; i < kFramesPerHop; i++)
                seg.feed(s, 0.f, (float)(fb + (uint64_t)i) * 0.01f);
            s->ml_frames.store(fb + kFramesPerHop);
            s->live_note.store(seg.cur_note);
            s->live_f0.store(0.f);
            continue;
        }

        // F0 tracker runs concurrently with the encoder — they're independent
        // and both leave >0 idle cores with spinning disabled. win16 is not
        // touched until both complete.
        auto f0_fut = std::async(std::launch::async, [&]() {
            std::vector<float> v = rmvpe.loaded()
                ? rmvpe.f0(win16.data() + kWinSamples16 - kRmvpeWin16, kRmvpeWin16)
                : yin_f0(win16.data(), kWinSamples16, kRate16);
            if (v.empty())
                v = yin_f0(win16.data(), kWinSamples16, kRate16);
            return v;
        });

        // HuBERT encode → phone features [T, 768]
        int T = 0;
        std::vector<float> hs = hubert.encode(win16.data(), kWinSamples16, &T);
        if (T < 1) { f0_fut.wait(); continue; }
        int D = (int)hs.size() / T;

        // First-frame diagnostic: show feature dim and ring fill
        static bool diag_logged = false;
        if (!diag_logged) {
            fprintf(stderr, "prism: D=%d ring=%zu\n",
                    D, s->output_ring.available());
            diag_logged = true;
        }

        // Repeat-interleave ×2: [T, D] → [2T, D]
        int T2 = T * 2;
        std::vector<float> phone_full(T2 * D);
        for (int t = 0; t < T; ++t) {
            const float* src = hs.data() + (size_t)t * D;
            std::copy(src, src + D, phone_full.data() + (size_t)(2*t)   * D);
            std::copy(src, src + D, phone_full.data() + (size_t)(2*t+1) * D);
        }

        // Slice to the last hop + crossfade (120ms): HuBERT gets full 400ms
        // context, VITS synthesises hop+20ms per call; the leading 20ms is
        // crossfaded against the previous call's tail.
        int n_vits = std::min(kSynFrames, T2);
        int t_off  = T2 - n_vits;
        std::vector<float> phone(phone_full.begin() + (size_t)t_off * D, phone_full.end());

        // F0 — RMVPE salience decode (YIN fallback), one value per 10ms frame,
        // computed concurrently above.
        std::vector<float> f0_frames = f0_fut.get();

        // Auto octave: keep a rolling history of the newly-arrived voiced
        // frames (window tail), re-evaluate {0,+12,-12} coverage against the
        // eroded register mask. A pending switch normally commits at the next
        // gated (silent) hop so the octave never jumps mid-phrase; an
        // overwhelming win (>0.25 coverage) commits immediately.
        int n_new = std::min((int)f0_frames.size(), kFramesPerHop);
        const float* tail = f0_frames.data() + (f0_frames.size() - (size_t)n_new);
        for (int i = 0; i < n_new; i++) {
            f0_hist[(size_t)hist_pos] = tail[i];
            hist_pos = (hist_pos + 1) % kHistLen;
        }
        hops_since_switch++;
        if (have_mask && s->auto_octave_on.load() && hops_since_switch >= 14) {
            float cur_cov = mask_coverage(core_mask, f0_hist.data(), kHistLen, cur_shift);
            int   best_shift = cur_shift;
            float best_cov   = cur_cov;
            for (int shift : {0, 12, -12}) {
                if (shift == cur_shift) continue;
                float c = mask_coverage(core_mask, f0_hist.data(), kHistLen, shift);
                if (c > best_cov + 0.10f) { best_cov = c; best_shift = shift; }
            }
            pending_shift = best_shift;
            if (pending_shift != cur_shift && best_cov - cur_cov > 0.25f) {
                cur_shift = pending_shift;    // clear win — worth the jump
                hops_since_switch = 0;
            }
        }
        if (!s->auto_octave_on.load()) { cur_shift = 0; pending_shift = 0; }
        s->auto_octave.store(cur_shift);

        float shift_mult = std::pow(2.f, cur_shift / 12.f);
        if (cur_shift != 0)
            for (auto& v : f0_frames) if (v > 0.f) v *= shift_mult;

        // Live transcription: feed the newly-arrived (post-shift) frames to the
        // note segmenter on the 10ms frame clock.
        uint64_t fbase = s->ml_frames.load();
        for (int i = 0; i < n_new; i++)
            seg.feed(s, tail[i], (float)(fbase + (uint64_t)i) * 0.01f);
        s->ml_frames.store(fbase + (uint64_t)n_new);
        s->live_note.store(seg.cur_note);
        s->live_f0.store(n_new > 0 ? tail[n_new - 1] : 0.f);

        // F0 for the synth window: RMVPE frames and interleaved phone frames
        // are both 10ms, so the last n_vits frames map 1:1 — no interpolation.
        std::vector<float> f0((size_t)n_vits, 0.f);
        for (int i = 0; i < n_vits && i < (int)f0_frames.size(); i++)
            f0[(size_t)(n_vits - 1 - i)] = f0_frames[f0_frames.size() - 1 - (size_t)i];

        if (s->easy_mode.load()) {
            for (auto& v : f0) {
                if (v > 50.f) {
                    if (smoothed_f0 < 50.f) smoothed_f0 = v;
                    smoothed_f0 = smoothed_f0 * 0.85f + v * 0.15f;
                    v = smoothed_f0;
                }
            }
        }

        // VITS inference
        std::vector<int64_t> sh_p   = {1, n_vits, D};
        std::vector<int64_t> sh_f   = {1, n_vits};
        std::vector<int64_t> sh_s   = {1};
        std::vector<int64_t> sh_rnd = {1, (int64_t)rnd_channels, n_vits};
        int64_t sid = 0;
        float phase_in = sine_phase;

        // Seeded Gaussian noise for VITS sampling (temperature 0.6666)
        std::vector<float> rnd_vec;
        std::vector<Ort::Value> ins;
        std::vector<const char*> in_names;

        ins.push_back(Ort::Value::CreateTensor<float>(
            mem, phone.data(), phone.size(), sh_p.data(), 3));
        in_names.push_back("phone");

        ins.push_back(Ort::Value::CreateTensor<float>(
            mem, f0.data(), f0.size(), sh_f.data(), 2));
        in_names.push_back("f0");

        if (has_rnd_in) {
            rnd_vec.resize((size_t)rnd_channels * n_vits);
            for (auto& v : rnd_vec) v = gauss(rng) * 0.66666f;
            ins.push_back(Ort::Value::CreateTensor<float>(
                mem, rnd_vec.data(), rnd_vec.size(), sh_rnd.data(), 3));
            in_names.push_back("rnd");
        }

        ins.push_back(Ort::Value::CreateTensor<int64_t>(
            mem, &sid, 1, sh_s.data(), 1));
        in_names.push_back("sid");

        if (has_phase_in) {
            ins.push_back(Ort::Value::CreateTensor<float>(
                mem, &phase_in, 1, sh_s.data(), 1));
            in_names.push_back("phase");
        }

        const char* out_names[] = {"audio"};
        std::vector<Ort::Value> result;
        try {
            result = vits->Run(Ort::RunOptions{nullptr},
                               in_names.data(), ins.data(), ins.size(),
                               out_names, 1);
        } catch (const std::exception& e) {
            snprintf(s->status_msg, sizeof(s->status_msg), "VITS err: %.100s", e.what());
            continue;
        }

        // Advance the oscillator phase across the emitted hop: each 10ms frame
        // adds f0/100 cycles (frame = sr/100 samples at rate sr).
        for (int i = 0; i < kFramesPerHop && i < n_vits; i++)
            sine_phase += f0[(size_t)i] / 100.f;
        sine_phase -= std::floor(sine_phase);

        float*  adata  = result[0].GetTensorMutableData<float>();
        int64_t M      = result[0].GetTensorTypeAndShapeInfo().GetShape()[2];

        // Resample model_sr → 48kHz
        if (out48.size() < (size_t)(M * 2)) out48.resize(M * 2);
        size_t n_out = resample_linear(adata, (size_t)M, out48.data(), model_sr, kRate48);
        for (size_t i = 0; i < n_out; ++i)
            out48[i] = std::max(-1.f, std::min(1.f, out48[i]));

        // Chunk layout after the guard skip: [hop 100ms][xfade tail 20ms].
        // Crossfade the hop's first 20ms against the previous call's tail,
        // emit the hop, save the two frames past it as the next tail.
        float* chunk = out48.data() + kGuard48;
        size_t usable = (n_out > (size_t)kGuard48) ? n_out - (size_t)kGuard48 : 0;
        size_t nxf = std::min((size_t)kXfade48, usable);
        for (size_t i = 0; i < nxf; ++i) {
            float a = (float)i / (float)kXfade48;
            chunk[i] = chunk[i] * a + xf_tail[i] * (1.f - a);
        }
        size_t emit = std::min(usable, (size_t)kHopSamples48);
        s->output_ring.push(chunk, emit);
        if (usable >= (size_t)(kHopSamples48 + kXfade48))
            std::memcpy(xf_tail.data(), chunk + kHopSamples48,
                        kXfade48 * sizeof(float));
        else
            std::memset(xf_tail.data(), 0, kXfade48 * sizeof(float));

        if (!s->vits_ring_ready.load()) {
            s->vits_ring_ready.store(true);
            snprintf(s->status_msg, sizeof(s->status_msg), "Voice ready");
        }

        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        s->ml_ms_per_chunk.store(ms);
        ms_ema = (ms_ema == 0.f) ? ms : ms_ema * 0.9f + ms * 0.1f;
        if (ms_ema > 115.f && s->vits_ring_ready.load())
            snprintf(s->status_msg, sizeof(s->status_msg),
                     "Can't keep up: %.0fms per 100ms chunk", ms_ema);

        // Bounded latency: if inference fell behind and input piled up, slide
        // the analysis window through the backlog without synthesising. The
        // output gets a dropout instead of an ever-growing delay.
        while (s->input_ring.available() >= (size_t)(kHopSamples48 * 2)) {
            s->input_ring.pull(hop48.data(), kHopSamples48);
            size_t nc = resample_linear(hop48.data(), kHopSamples48,
                                        hop16.data(), kRate48, kRate16);
            std::memmove(win16.data(), win16.data() + nc,
                         (kWinSamples16 - nc) * sizeof(float));
            std::memcpy(win16.data() + (kWinSamples16 - nc), hop16.data(),
                        nc * sizeof(float));
        }
    }

    // Close any note left open when the thread stops
    seg.close(s, (float)s->ml_frames.load() * 0.01f);
    s->live_note.store(-1);
    s->live_f0.store(0.f);
}

void ml_thread_start(AppState* s) {
    ml_thread_stop(s);  // join any previous thread before spawning a new one
    s->ml_running.store(true);
    s->ml_thread_handle = std::thread(ml_thread_main, s);
}

void ml_thread_stop(AppState* s) {
    s->ml_running.store(false);
    if (s->ml_thread_handle.joinable())
        s->ml_thread_handle.join();
    // Silence output during the gap before the new thread primes its ring.
    // Don't touch read_pos — that races with the audio callback's pull().
    s->vits_ring_ready.store(false);
    s->crossfade = 0.f;
    s->vits_fade  = 0.f;
}
