#include "prism.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

void audio_callback(ma_device* dev, void* out, const void* in, ma_uint32 frames) {
    AppState*    s    = (AppState*)dev->pUserData;
    const float* inp  = (const float*)in;
    float*       outp = (float*)out;

    // Push left-channel mono to InputRing for the ML thread
    {
        float mono[4096];
        ma_uint32 n = frames < 4096 ? frames : 4096;
        for (ma_uint32 i = 0; i < n; ++i)
            mono[i] = inp[i * 2];
        s->input_ring.push(mono, n);
    }

    float mix = s->param_mix.load(std::memory_order_relaxed);

    if (s->rvc_loaded.load(std::memory_order_acquire)) {
        // RVC mode: mix synthesized audio from ML thread with dry mic signal
        for (ma_uint32 i = 0; i < frames; ++i) {
            float mic = inp[i * 2];
            float synth = 0.f;
            s->output_ring.pull(&synth, 1);
            float wet = mic + (synth - mic) * mix;
            outp[i * 2]     = wet;
            outp[i * 2 + 1] = wet;
        }
    } else {
        // Dry passthrough
        for (ma_uint32 i = 0; i < frames; ++i) {
            float x = inp[i * 2];
            outp[i * 2]     = x;
            outp[i * 2 + 1] = x;
        }
    }
}

bool audio_init(AppState* s, ma_device* dev) {
    ma_device_config cfg = ma_device_config_init(ma_device_type_duplex);
    cfg.capture.format    = ma_format_f32;
    cfg.capture.channels  = 2;
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = 2;
    cfg.sampleRate        = 44100;
    cfg.dataCallback      = audio_callback;
    cfg.pUserData         = s;

    if (ma_device_init(nullptr, &cfg, dev) != MA_SUCCESS)
        return false;
    s->audio_device = dev;
    return true;
}

void audio_start(ma_device* dev, AppState* s) {
    ma_device_start(dev);
    s->audio_active.store(true);
}

void audio_stop(ma_device* dev, AppState* s) {
    ma_device_stop(dev);
    s->audio_active.store(false);
}
