#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "spsc_queue.h"
#include "input_ring.h"
#include "biquad.h"
#include "../vendor/grain_shifter.h"

// Forward declarations
struct ma_device;
namespace Ort { class Session; class Env; }

struct AppState {
    // Parameters (written by UI thread, read by audio/ML threads)
    std::atomic<float> param_blend{0.5f};          // LPC warp intensity 0–1
    std::atomic<float> param_formant_shift{0.0f};  // semitones -2 to +2
    std::atomic<float> param_smoothing{0.4f};      // coefficient chase rate 0–1
    std::atomic<float> param_voiced_thresh{0.5f};  // YIN confidence cutoff 0–1
    std::atomic<float> param_mix{0.85f};            // wet/dry 0–1

    // Lock-free communication between threads
    SpscQueue<64>  coeff_queue;
    InputRing      input_ring;
    CoeffPacket    last_packet{};

    // DSP state (audio thread only)
    BiquadCascade cascade[2];
    GrainShifter  shifter[2];

    // Model paths (written once by UI before ML thread starts)
    std::string model_path;    // .pth file
    std::string onnx_path;     // exported conv encoder .onnx
    std::string speaker_path;  // target speaker .wav

    // Target speaker state (computed at load, read-only in ML thread)
    float target_lpc[12]{};
    float target_f0_mean = 0.0f;
    bool  target_loaded = false;

    // ML thread control
    std::atomic<bool> ml_running{false};
    std::atomic<float> ml_ms_per_chunk{0.0f};
    std::atomic<float> ml_voiced_pct{0.0f};

    // Audio device (owned by main thread)
    ma_device* audio_device = nullptr;
    std::atomic<bool> audio_active{false};

    // Export thread state (written by export thread, read by UI thread)
    std::atomic<int> export_status{0};  // 0=idle 1=running 2=done 3=error
    std::string      export_error;

    // UI state
    bool model_loaded = false;
    char speaker_path_buf[512]{};
    char status_msg[128] = "Ready";
};
