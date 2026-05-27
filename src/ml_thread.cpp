#include "prism.h"
#include "lpc.h"
#include "yin.h"

struct PrismModel;

#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

static constexpr int kOrder    = 12;
static constexpr int kChunk    = 1764;   // 40ms @ 44.1k

static constexpr float kSampleRate = 44100.0f;

static void ml_thread_main(AppState* s, PrismModel* model) {
    float chunk_buf[kChunk];
    float autocorr[kOrder + 1];
    float lpc_src[kOrder];
    float lpc_warp[kOrder];

    YinDetector yin;
    yin.init(kSampleRate);

    // Warm-up: wait until enough samples are buffered
    while (s->ml_running.load()) {
        if (s->input_ring.available() >= kChunk)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    while (s->ml_running.load()) {
        if (s->input_ring.available() < kChunk) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        auto t_start = std::chrono::steady_clock::now();

        s->input_ring.pull(chunk_buf, kChunk);

        // Source LPC from direct audio autocorrelation
        for (int lag = 0; lag <= kOrder; ++lag) {
            float sum = 0.0f;
            for (int i = 0; i < kChunk - lag; ++i)
                sum += chunk_buf[i] * chunk_buf[i + lag];
            autocorr[lag] = (lag == 0) ? sum + 1e-6f : sum;
        }
        levinson_durbin(autocorr, kOrder, lpc_src);

        // Warp toward target speaker
        float blend = s->param_blend.load();
        if (s->target_loaded) {
            for (int i = 0; i < kOrder; ++i)
                lpc_warp[i] = lpc_src[i] + (s->target_lpc[i] - lpc_src[i]) * blend;
        } else {
            for (int i = 0; i < kOrder; ++i)
                lpc_warp[i] = lpc_src[i];
        }

        // Formant shift: scale LPC coefficients by semitone ratio
        float semitones = s->param_formant_shift.load();
        if (std::fabs(semitones) > 0.01f) {
            float ratio = std::pow(2.0f, semitones / 12.0f);
            float r_k = ratio;
            for (int k = 0; k < kOrder; ++k) {
                lpc_warp[k] *= r_k;
                r_k *= ratio;
            }
        }

        // YIN pitch detection on raw chunk
        for (int i = 0; i < kChunk; ++i) yin.push_sample(chunk_buf[i]);
        if (yin.pending) yin.run_detect();
        float src_f0   = yin.pitch_hz;
        float voiced   = yin.confidence;
        s->ml_voiced_pct.store(voiced);

        // Pitch ratio toward target speaker F0
        float pitch_ratio = 1.0f;
        if (s->target_loaded && src_f0 > 50.0f && voiced > s->param_voiced_thresh.load())
            pitch_ratio = std::clamp(s->target_f0_mean / src_f0, 0.5f, 2.0f);

        // Build CoeffPacket
        CoeffPacket pkt{};
        float gain = 1.0f;
        lpc_to_biquad(lpc_warp, kOrder, pkt.biquad, gain);
        pkt.pitch_ratio = pitch_ratio;
        pkt.voiced      = voiced;

        s->coeff_queue.push(pkt);

        auto t_end = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        s->ml_ms_per_chunk.store(ms);

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void ml_thread_start(AppState* s, PrismModel* model) {
    s->ml_running.store(true);
    std::thread(ml_thread_main, s, model).detach();
}

void ml_thread_stop(AppState* s) {
    s->ml_running.store(false);
}
