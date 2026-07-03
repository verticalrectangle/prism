#pragma once

#include <string>
#include <vector>

struct AppState;
struct ma_device;
struct ma_context;

struct AudioDevice {
    std::string id;    // PipeWire/PA device name (empty = system default)
    std::string name;  // human-readable label
};

struct AudioDeviceList {
    std::vector<AudioDevice> playback;
    std::vector<AudioDevice> capture;
};

// Fill list — call once at startup with an initialised ma_context
void audio_enumerate(ma_context* ctx, AudioDeviceList& out);

bool audio_init(AppState* s, ma_device* dev,
                const std::string& capture_id,
                const std::string& playback_id);

void audio_start(ma_device* dev, AppState* s);
void audio_stop(ma_device* dev, AppState* s);
void audio_uninit(ma_device* dev);   // no-op on the PipeWire backend

bool        audio_is_pipewire();
const char* audio_capture_name(const ma_device* dev);
const char* audio_playback_name(const ma_device* dev);

// Blend the output-ring voice with the dry signal into out[0..n). Runs on
// the RT audio thread of whichever backend is active.
void audio_render_block(AppState* s, float* out, const float* dry, unsigned n);
