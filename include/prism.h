#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "input_ring.h"
#include "output_ring.h"

struct ma_device;

struct AppState {
    // Parameters (written by UI thread)
    std::atomic<float> param_formant_shift{0.0f};  // f0 semitone shift -2 to +2
    std::atomic<float> param_mix{0.85f};            // wet/dry 0–1

    // Lock-free audio pipeline
    InputRing  input_ring;    // mic audio → ML thread (44.1 kHz mono)
    OutputRing output_ring;   // synthesized audio → audio thread (44.1 kHz mono)

    // Model state (set by UI/export threads, read by ML thread)
    std::atomic<bool> hubert_loaded{false};
    std::atomic<bool> rvc_loaded{false};
    int rvc_sr{40000};        // from sidecar JSON after RVC export
    int rvc_phone_dim{768};   // from sidecar JSON after RVC export

    // ML thread control
    std::atomic<bool>  ml_running{false};
    std::atomic<float> ml_ms_per_chunk{0.0f};

    // Active RVC model path (used to show "Selected" state in UI)
    std::string rvc_model_path;

    // Audio device (owned by main thread)
    ma_device* audio_device = nullptr;
    std::atomic<bool> audio_active{false};

    // Export/HuBERT-load thread state
    std::atomic<int> export_status{0};   // 0=idle 1=running 2=done 3=error
    std::string      export_error;

    std::atomic<int> hubert_init_status{0};  // same 0/1/2/3 encoding
    std::string      hubert_init_error;

    // UI status line
    char status_msg[128] = "Loading HuBERT...";
};
