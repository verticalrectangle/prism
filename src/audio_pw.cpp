#ifdef HAVE_PIPEWIRE
#include "audio_pw.h"
#include "prism.h"
#include "audio.h"

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>

#include <cstring>
#include <ctime>

static AppState*       g_s    = nullptr;
static pw_thread_loop* g_loop = nullptr;
static pw_stream*      g_out  = nullptr;
static pw_stream*      g_in   = nullptr;
static bool            g_on   = false;

// Capture → playback dry-monitor path. Both streams tick the same graph
// clock, so this stays one cycle deep after the initial prime.
static InputRing g_dry;

static void on_in_process(void*) {
    pw_buffer* b = pw_stream_dequeue_buffer(g_in);
    if (!b) return;
    spa_data& d = b->buffer->datas[0];
    if (d.data && d.chunk && d.chunk->size > 0) {
        const uint32_t stride = d.chunk->stride > 0 ? (uint32_t)d.chunk->stride
                                                    : sizeof(float);
        const float* src = (const float*)((const uint8_t*)d.data + d.chunk->offset);
        uint32_t frames = d.chunk->size / stride;
        g_s->input_ring.push(src, frames);
        g_dry.push(src, frames);
    }
    pw_stream_queue_buffer(g_in, b);
}

static void on_out_process(void*) {
    pw_buffer* b = pw_stream_dequeue_buffer(g_out);
    if (!b) return;
    spa_data& d = b->buffer->datas[0];
    float* dst = (float*)d.data;
    if (!dst) { pw_stream_queue_buffer(g_out, b); return; }
    const uint32_t stride = sizeof(float);
    uint32_t max_frames = d.maxsize / stride;
    uint32_t frames = b->requested ? (uint32_t)b->requested : max_frames;
    if (frames > max_frames) frames = max_frames;

    float dry[512];
    uint32_t done = 0;
    while (done < frames) {
        uint32_t n = frames - done;
        if (n > 512) n = 512;
        size_t av   = g_dry.available();
        size_t take = av < n ? av : n;
        if (take) g_dry.pull(dry, take);
        if (take < n)
            std::memset(dry + take, 0, (n - take) * sizeof(float));
        audio_render_block(g_s, dst + done, dry, n);
        done += n;
    }
    d.chunk->offset = 0;
    d.chunk->stride = (int32_t)stride;
    d.chunk->size   = frames * stride;
    pw_stream_queue_buffer(g_out, b);
}

// pw_stream_events has a long member list; C++ can't partially designate, so
// fill the two we need at runtime.
static pw_stream_events g_out_events;
static pw_stream_events g_in_events;

static const spa_pod* mono_48k_format(spa_pod_builder* pb) {
    spa_audio_info_raw info;
    memset(&info, 0, sizeof(info));
    info.format      = SPA_AUDIO_FORMAT_F32;
    info.rate        = 48000;
    info.channels    = 1;
    info.position[0] = SPA_AUDIO_CHANNEL_MONO;
    return spa_format_audio_raw_build(pb, SPA_PARAM_EnumFormat, &info);
}

bool audio_pw_start(AppState* s) {
    if (g_on) return true;
    g_s = s;

    // One quantum of silence absorbs in/out scheduling jitter at startup.
    float zeros[128] = {};
    g_dry.push(zeros, 128);

    pw_init(nullptr, nullptr);
    g_loop = pw_thread_loop_new("prism-audio", nullptr);
    if (!g_loop) return false;
    if (pw_thread_loop_start(g_loop) != 0) {
        pw_thread_loop_destroy(g_loop);
        g_loop = nullptr;
        return false;
    }

    memset(&g_out_events, 0, sizeof(g_out_events));
    g_out_events.version = PW_VERSION_STREAM_EVENTS;
    g_out_events.process = on_out_process;
    memset(&g_in_events, 0, sizeof(g_in_events));
    g_in_events.version = PW_VERSION_STREAM_EVENTS;
    g_in_events.process = on_in_process;

    pw_thread_loop_lock(g_loop);

    const auto flags = (pw_stream_flags)(PW_STREAM_FLAG_AUTOCONNECT |
                                         PW_STREAM_FLAG_MAP_BUFFERS |
                                         PW_STREAM_FLAG_RT_PROCESS);
    uint8_t podbuf[1024];

    g_out = pw_stream_new_simple(
        pw_thread_loop_get_loop(g_loop), "Prism",
        pw_properties_new(PW_KEY_MEDIA_TYPE,     "Audio",
                          PW_KEY_MEDIA_CATEGORY, "Playback",
                          PW_KEY_MEDIA_ROLE,     "Production",
                          PW_KEY_NODE_LATENCY,   "128/48000",
                          nullptr),
        &g_out_events, nullptr);
    if (g_out) {
        spa_pod_builder pb = SPA_POD_BUILDER_INIT(podbuf, sizeof(podbuf));
        const spa_pod* params[1] = { mono_48k_format(&pb) };
        if (pw_stream_connect(g_out, PW_DIRECTION_OUTPUT, PW_ID_ANY,
                              flags, params, 1) < 0) {
            pw_stream_destroy(g_out);
            g_out = nullptr;
        }
    }

    g_in = pw_stream_new_simple(
        pw_thread_loop_get_loop(g_loop), "Prism Mic",
        pw_properties_new(PW_KEY_MEDIA_TYPE,     "Audio",
                          PW_KEY_MEDIA_CATEGORY, "Capture",
                          PW_KEY_MEDIA_ROLE,     "Production",
                          PW_KEY_NODE_LATENCY,   "128/48000",
                          nullptr),
        &g_in_events, nullptr);
    if (g_in) {
        spa_pod_builder pb = SPA_POD_BUILDER_INIT(podbuf, sizeof(podbuf));
        const spa_pod* params[1] = { mono_48k_format(&pb) };
        if (pw_stream_connect(g_in, PW_DIRECTION_INPUT, PW_ID_ANY,
                              flags, params, 1) < 0) {
            pw_stream_destroy(g_in);
            g_in = nullptr;
        }
    }

    pw_thread_loop_unlock(g_loop);

    if (!g_out || !g_in) {
        audio_pw_stop();
        return false;
    }

    // Wait (bounded) for both streams to reach STREAMING — an error or a
    // timeout means PipeWire isn't going to schedule us; fall back.
    for (int i = 0; i < 100; ++i) {  // ≤ 2 s
        pw_thread_loop_lock(g_loop);
        pw_stream_state so = pw_stream_get_state(g_out, nullptr);
        pw_stream_state si = pw_stream_get_state(g_in, nullptr);
        pw_thread_loop_unlock(g_loop);
        if (so == PW_STREAM_STATE_ERROR || si == PW_STREAM_STATE_ERROR) {
            audio_pw_stop();
            return false;
        }
        if (so == PW_STREAM_STATE_STREAMING && si == PW_STREAM_STATE_STREAMING) {
            g_on = true;
            return true;
        }
        struct timespec ts = {0, 20 * 1000 * 1000};
        nanosleep(&ts, nullptr);
    }
    audio_pw_stop();
    return false;
}

void audio_pw_stop() {
    if (!g_loop) return;
    pw_thread_loop_lock(g_loop);
    if (g_in)  { pw_stream_destroy(g_in);  g_in  = nullptr; }
    if (g_out) { pw_stream_destroy(g_out); g_out = nullptr; }
    pw_thread_loop_unlock(g_loop);
    pw_thread_loop_stop(g_loop);
    pw_thread_loop_destroy(g_loop);
    g_loop = nullptr;
    g_on   = false;
}

bool audio_pw_active() { return g_on; }
#endif  // HAVE_PIPEWIRE
