#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// Streaming wav2vec2 inference. Expects a dual-output ONNX:
//   input:  "input_values" [1, N] float32  (16 kHz mono)
//   output: "hidden_states" [1, T, 768] float32  (final transformer layer)
//           "logits"        [1, T, V]   float32  (CTC logits, optional)
// T ≈ N / 320  (wav2vec2 conv stride = 320 samples at 16 kHz)
struct Wav2Vec2Encoder {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "wav2vec2"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    bool loaded = false;
    bool has_hidden_states = false;
    bool needs_attention_mask = false;

    void load(const std::string& onnx_path, int n_threads = 2);
    bool is_loaded() const { return loaded; }

    // Encode n_samples of 16 kHz mono audio.
    // Returns hidden_states flattened [T * 768], sets *T_out = T.
    // Falls back to logits if hidden_states output not present.
    std::vector<float> encode(const float* audio16k, int n_samples, int* T_out);
};
