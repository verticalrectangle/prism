#include "prism.h"
#include "audio.h"
#include "audio_pw.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <algorithm>

static AppState* g_state  = nullptr;
static bool      g_use_pw = false;
static bool      g_ma_init = false;

// Shared by the miniaudio duplex callback and the PipeWire output callback:
// blend the converted voice from the output ring with the dry signal.
void audio_render_block(AppState* s, float* out, const float* dry, unsigned n) {
    float vits[512];
    size_t got = (n <= 512) ? s->output_ring.pull(vits, n) : 0;
    float  mix = s->param_mix.load();

    static constexpr float kWarmStep = 1.f / 960.f;  // 20ms @ 48kHz
    static constexpr float kFadeStep = 1.f / 48.f;   // 1ms  @ 48kHz

    bool warming = s->vits_ring_ready.load() && s->crossfade < 1.f;
    float cv = s->crossfade;
    float vf = s->vits_fade;

    for (unsigned i = 0; i < n; ++i) {
        if (warming) cv = std::min(1.f, cv + kWarmStep);

        bool have = (i < (unsigned)got);
        vf = have ? std::min(1.f, vf + kFadeStep)
                  : std::max(0.f, vf - kFadeStep);

        float vits_sig = have ? vits[i] * mix : 0.f;
        float blended  = vits_sig * vf + dry[i] * (1.f - vf);
        out[i]         = dry[i] * (1.f - cv) + blended * cv;
    }
    s->crossfade = cv;
    s->vits_fade = vf;
}

static void prism_audio_callback(ma_device* /*dev*/, void* out_buf, const void* in_buf, ma_uint32 n) {
    AppState* s = g_state;
    const float* in  = (const float*)in_buf;
    float*       out = (float*)out_buf;

    // Push mono mic to ML input ring
    s->input_ring.push(in, n);
    audio_render_block(s, out, in, n);
}

void audio_enumerate(ma_context* ctx, AudioDeviceList& list) {
    list.playback.clear();
    list.capture.clear();
    list.playback.push_back({"", "System default"});
    list.capture.push_back({"", "System default"});

    ma_device_info* pb; ma_uint32 pb_n = 0;
    ma_device_info* cap; ma_uint32 cap_n = 0;
    if (ma_context_get_devices(ctx, &pb, &pb_n, &cap, &cap_n) != MA_SUCCESS)
        return;
    for (ma_uint32 i = 0; i < pb_n;  ++i) list.playback.push_back({pb[i].name,  pb[i].name});
    for (ma_uint32 i = 0; i < cap_n; ++i) list.capture.push_back({cap[i].name, cap[i].name});
}

bool audio_init(AppState* s, ma_device* dev,
                const std::string& /*capture_id*/,
                const std::string& /*playback_id*/) {
    g_state = s;

#ifdef HAVE_PIPEWIRE
    // Native PipeWire first: real 128-frame quantum, capture and playback on
    // the same graph clock — the dry monitor path is ~2.7ms instead of the
    // pulse shim's ~14ms round trip.
    if (audio_pw_start(s)) {
        g_use_pw = true;
        fprintf(stderr, "Audio: native PipeWire duplex @ 48kHz, 128-frame quantum\n");
        s->audio_device = dev;
        return true;
    }
    fprintf(stderr, "Audio: PipeWire unavailable, falling back to miniaudio\n");
#endif

    ma_device_config cfg      = ma_device_config_init(ma_device_type_duplex);
    cfg.capture.format        = ma_format_f32;
    cfg.capture.channels      = 1;
    cfg.playback.format       = ma_format_f32;
    cfg.playback.channels     = 1;
    cfg.sampleRate            = 48000;
    cfg.periodSizeInFrames    = 128;
    cfg.dataCallback          = prism_audio_callback;
    // pUserData unused — using global g_state like silvertune

    if (ma_device_init(nullptr, &cfg, dev) != MA_SUCCESS)
        return false;
    g_ma_init = true;

    fprintf(stderr, "Audio: %s → %s @ %uHz %uch\n",
            dev->capture.name, dev->playback.name,
            dev->sampleRate, dev->capture.channels);

    s->audio_device = dev;
    return true;
}

bool audio_is_pipewire() { return g_use_pw; }

const char* audio_capture_name(const ma_device* dev) {
    return g_use_pw ? "PipeWire (default source)" : dev->capture.name;
}
const char* audio_playback_name(const ma_device* dev) {
    return g_use_pw ? "PipeWire (default sink)" : dev->playback.name;
}

void audio_start(ma_device* dev, AppState* s) {
    if (!g_use_pw) ma_device_start(dev);   // PipeWire streams already run
    s->audio_active.store(true);
}

void audio_stop(ma_device* dev, AppState* s) {
#ifdef HAVE_PIPEWIRE
    if (g_use_pw) audio_pw_stop();
#endif
    if (g_ma_init) ma_device_stop(dev);
    s->audio_active.store(false);
}

void audio_uninit(ma_device* dev) {
    if (g_ma_init) { ma_device_uninit(dev); g_ma_init = false; }
}
