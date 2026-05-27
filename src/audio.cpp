#include "prism.h"
#include "yin.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <cstring>

void audio_callback(ma_device* dev, void* out, const void* in, ma_uint32 frames) {
    AppState* s = (AppState*)dev->pUserData;

    CoeffPacket pkt;
    if (s->coeff_queue.pop(pkt)) {
        for (int ch = 0; ch < 2; ++ch)
            s->cascade[ch].set_coeffs(pkt.biquad, static_cast<int>(frames));
        s->last_packet = pkt;
    }

    const float* inp  = (const float*)in;
    float*       outp = (float*)out;

    // Mono mix into InputRing for ML thread (left channel only)
    s->input_ring.push(inp, frames);

    float mix    = s->param_mix.load();
    float voiced = s->last_packet.voiced;
    float thresh = s->param_voiced_thresh.load();

    for (ma_uint32 i = 0; i < frames; ++i) {
        float x = inp[i * 2];  // mono from left channel

        float y = s->shifter[0].process(x, static_cast<double>(s->last_packet.pitch_ratio));
        y = (voiced >= thresh) ? s->cascade[0].process(y) : x;

        float wet = x + (y - x) * mix;
        outp[i * 2]     = wet;
        outp[i * 2 + 1] = wet;
    }
}

bool audio_init(AppState* s, ma_device* dev) {
    ma_device_config cfg = ma_device_config_init(ma_device_type_duplex);
    cfg.capture.format   = ma_format_f32;
    cfg.capture.channels = 2;
    cfg.playback.format  = ma_format_f32;
    cfg.playback.channels = 2;
    cfg.sampleRate       = 44100;
    cfg.dataCallback     = audio_callback;
    cfg.pUserData        = s;

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
