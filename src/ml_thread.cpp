#include "prism.h"
#include "model.h"
#include "lpc.h"

#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

static constexpr int kOrder    = 12;
static constexpr int kChunk    = 1764;   // 40ms @ 44.1k

static void ml_thread_main(AppState* s, PrismModel* model) {
    float chunk_buf[kChunk];
    float autocorr[kOrder + 1];
    float lpc_src[kOrder];
    float lpc_warp[kOrder];

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

        // HuBERT conv features
        const float* feat = model->infer(chunk_buf, kChunk);

        // Source LPC from features
        features_to_autocorr(feat, kFeatDim, autocorr, kOrder);
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

        // Build CoeffPacket
        CoeffPacket pkt{};
        float gain = 1.0f;
        lpc_to_biquad(lpc_warp, kOrder, pkt.biquad, gain);
        pkt.pitch_ratio = 1.0f;  // pitch handled by grain shifter based on voiced
        pkt.voiced = 0.5f;       // placeholder — YIN runs in audio thread

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
