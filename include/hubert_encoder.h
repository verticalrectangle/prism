#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// Streaming HuBERT (ContentVec) inference — the standard RVC content encoder.
// Expects a single-input single-output ONNX:
//   input:  "audio"   [1, N] float32  (16 kHz mono, raw — no normalization)
//   output: "features" [1, T, 768] float32  (phone features, 10ms stride)
// T ≈ N / 320  (conv stride = 320 samples at 16 kHz)
struct HubertEncoder {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "hubert"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    bool loaded = false;

    void load(const std::string& onnx_path, int n_threads = 2);
    bool is_loaded() const { return loaded; }

    // Encode n_samples of 16 kHz mono audio (raw, no preprocessing needed).
    // Returns features flattened [T * 768], sets *T_out = T.
    std::vector<float> encode(const float* audio16k, int n_samples, int* T_out);
};
