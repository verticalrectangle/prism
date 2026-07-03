#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "input_ring.h"
#include "output_ring.h"

// Forward declarations
struct ma_device;

// One transcribed note, times in seconds on the ML thread's frame clock
// (10ms RMVPE frames since the ML thread started).
struct NoteEvent {
    float t_on  = 0.f;
    float t_off = 0.f;
    int   note  = 0;    // MIDI note number
};

struct AppState {
    // Parameters (written by UI thread, read by ML thread)
    std::atomic<float> param_mix{1.0f};       // output level 0–1
    std::atomic<bool>  easy_mode{false};       // pitch contour smoothing
    std::atomic<bool>  auto_octave_on{true};   // register-mask driven octave shift
    std::atomic<float> gate_threshold{0.006f}; // silence gate open RMS (close = /2)

    // Gate telemetry (ML thread writes, UI reads)
    std::atomic<float> input_rms{0.f};         // per-hop input level
    std::atomic<bool>  gate_open{false};

    // Lock-free audio I/O between threads
    InputRing  input_ring;   // audio thread → ML thread (44.1kHz mono)
    OutputRing output_ring;  // ML thread → audio thread (44.1kHz mono)

    // Model state (written once by use_model, read by ML thread)
    std::string vits_onnx_path;
    std::atomic<int>  rvc_version{0};    // 0=none, 1=v1, 2=v2
    std::atomic<int>  output_sr{40000};  // model native sample rate
    std::atomic<bool> vits_loaded{false};
    std::string model_path;

    // Trained-register mask from the sidecar JSON (see rvc_onnx.cpp):
    // which of the 256 coarse-pitch bins the voice model was fine-tuned on.
    // Written by use_model before the ML thread starts.
    std::atomic<bool> have_register_mask{false};
    uint8_t register_mask[256] = {};

    // ML thread control
    std::atomic<bool>  ml_running{false};
    std::atomic<float> ml_ms_per_chunk{0.0f};
    std::atomic<bool>  rmvpe_active{false};   // neural pitch vs YIN fallback
    std::thread          ml_thread_handle;   // joinable handle (UI thread only)
    // Model lifecycle for UI: 0=idle 1=exporting 2=loading 3=ready 4=error
    std::atomic<int>     model_state{0};
    std::string          loading_path;       // path being loaded (UI thread only)
    std::atomic<uint64_t> export_gen{0};     // supersedes stale export threads

    // Live pitch transcription (ML thread writes, UI reads)
    std::atomic<int>      auto_octave{0};     // applied shift, semitones
    std::atomic<float>    live_f0{0.f};       // post-shift f0 in Hz, 0 = unvoiced
    std::atomic<int>      live_note{-1};      // current MIDI note, -1 = silence
    std::atomic<uint64_t> ml_frames{0};       // 10ms frames since ML start
    std::mutex            notes_mu;           // guards notes (ML + UI threads)
    std::vector<NoteEvent> notes;             // completed notes, oldest first

    // Export thread state
    std::atomic<int> export_status{0};  // 0=idle 1=running 2=done 3=error
    std::string      export_error;

    // Audio device (owned by main thread)
    ma_device* audio_device = nullptr;
    std::atomic<bool> audio_active{false};

    // Warm-up / crossfade — audio thread only, no atomics needed
    std::atomic<bool> vits_ring_ready{false};
    float crossfade  = 0.f;  // warmup: ramps 0→1 once ring is primed
    float vits_fade  = 0.f;  // per-block: ramps 0→1 when ring has data, back when dry

    // UI state
    char status_msg[128] = "Ready";
};
