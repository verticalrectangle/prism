#include "prism.h"
#include "model.h"
#include "ml_thread.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int   kChunk44  = 8820;    // 200ms at 44100 Hz
static constexpr int   kChunk16  = 3200;    // ~200ms at 16000 Hz (3200/16000 = 200ms)
static constexpr float kRate44   = 44100.f;
static constexpr float kRate16   = 16000.f;

// ── Batch YIN pitch detector ──────────────────────────────────────────────────
// Returns the fundamental frequency in Hz for audio window x[0..n-1].
// Returns 0 if unvoiced (no clear periodicity found).
static float yin_pitch(const float* x, int n, float sr) {
    int tau_max = std::min(n / 2 - 1, (int)(sr / 50.f));
    int tau_min = std::max(2,          (int)(sr / 500.f));
    if (tau_max <= tau_min) return 0.f;

    std::vector<float> d(tau_max + 1, 0.f);
    for (int tau = 1; tau <= tau_max; tau++) {
        int len = n - tau;
        for (int i = 0; i < len; i++) {
            float diff = x[i] - x[i + tau];
            d[tau] += diff * diff;
        }
    }

    d[0] = 1.f;
    float sum = 0.f;
    for (int tau = 1; tau <= tau_max; tau++) {
        sum += d[tau];
        d[tau] = (sum > 0.f) ? d[tau] * (float)tau / sum : 1.f;
    }

    static constexpr float kThresh = 0.12f;
    for (int tau = tau_min; tau <= tau_max; tau++) {
        if (d[tau] < kThresh) {
            if (tau > 1 && tau < tau_max) {
                float s0 = d[tau - 1], s1 = d[tau], s2 = d[tau + 1];
                float denom = 2.f * s1 - s0 - s2;
                float adj = (std::fabs(denom) > 1e-6f) ? 0.5f * (s2 - s0) / denom : 0.f;
                return sr / ((float)tau + adj);
            }
            return sr / (float)tau;
        }
    }
    return 0.f;
}

// Compute per-frame f0 at n_frames evenly-spaced positions across x[0..n-1].
static std::vector<float> compute_f0_frames(const float* x, int n, float sr, int n_frames) {
    std::vector<float> f0(n_frames, 0.f);
    if (n_frames <= 0 || n <= 0) return f0;
    int win = std::min(n, 512);  // 32ms at 16kHz
    for (int i = 0; i < n_frames; i++) {
        int center = (int)((i + 0.5f) * n / n_frames);
        int start  = center - win / 2;
        start = std::max(0, std::min(start, n - win));
        f0[i] = yin_pitch(x + start, win, sr);
    }
    return f0;
}

// ── ML thread ─────────────────────────────────────────────────────────────────

static void ml_thread_main(AppState* s, HubertModel* hubert, RvcModel* rvc) {
    std::vector<float> chunk44(kChunk44);
    std::vector<float> chunk16(kChunk16 + 4, 0.f);
    std::vector<float> resamp_out(kChunk44 + 4096, 0.f);

    // Wait until the InputRing has enough audio buffered
    while (s->ml_running.load()) {
        if (s->input_ring.available() >= (size_t)kChunk44) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    while (s->ml_running.load()) {
        if (s->input_ring.available() < (size_t)kChunk44) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();

        s->input_ring.pull(chunk44.data(), kChunk44);

        // Resample 44100 Hz → 16000 Hz for HuBERT
        int n16k = resample_linear(chunk44.data(), kChunk44, kRate44,
                                   chunk16.data(), kChunk16 + 4, kRate16);

        // HuBERT inference → features [T_conv, 768]
        int T_conv = 0;
        std::vector<float> feat = hubert->infer(chunk16.data(), n16k, &T_conv);
        if (T_conv == 0 || feat.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 2× frame repeat: [T_conv, 768] → [T2, phone_dim]
        // This doubles the temporal resolution to match the RVC decoder's expected rate.
        int T2    = T_conv * 2;
        int pdim  = rvc->phone_dim;

        std::vector<float> phone((size_t)T2 * pdim);
        for (int t = 0; t < T_conv; t++) {
            const float* src = feat.data() + (size_t)t * 768;
            float* d0 = phone.data() + (size_t)(2 * t)     * pdim;
            float* d1 = phone.data() + (size_t)(2 * t + 1) * pdim;
            std::copy(src, src + pdim, d0);
            std::copy(src, src + pdim, d1);
        }

        // Per-frame f0 on the 16kHz chunk
        auto f0 = compute_f0_frames(chunk16.data(), n16k, kRate16, T2);

        // Apply semitone pitch shift
        float semitones = s->param_formant_shift.load();
        if (std::fabs(semitones) > 0.01f) {
            float mult = std::pow(2.f, semitones / 12.f);
            for (auto& v : f0)
                if (v > 50.f) v *= mult;
        }

        // RVC inference → audio [M] at rvc->sr
        std::vector<float> audio_rvc = rvc->infer(phone.data(), T2, f0.data());
        if (audio_rvc.empty()) continue;

        // Resample rvc->sr → 44100 Hz
        int n_out = resample_linear(audio_rvc.data(), (int)audio_rvc.size(),
                                    (float)rvc->sr,
                                    resamp_out.data(), (int)resamp_out.size(),
                                    kRate44);

        // Push synthesized audio to output ring
        s->output_ring.push(resamp_out.data(), (size_t)n_out);

        auto t1 = std::chrono::steady_clock::now();
        s->ml_ms_per_chunk.store(
            std::chrono::duration<float, std::milli>(t1 - t0).count());
    }
}

void ml_thread_start(AppState* s, HubertModel* hubert, RvcModel* rvc) {
    s->ml_running.store(true);
    std::thread(ml_thread_main, s, hubert, rvc).detach();
}

void ml_thread_stop(AppState* s) {
    s->ml_running.store(false);
}
