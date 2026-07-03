#include "wav2vec2_encoder.h"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <numeric>

void Wav2Vec2Encoder::load(const std::string& onnx_path, int n_threads) {
    opts.SetIntraOpNumThreads(n_threads > 0 ? n_threads : 2);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // Idle ORT threads spin-wait by default; with several sessions in one
    // process the spinners steal CPU from whichever session is running.
    opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
    session = std::make_unique<Ort::Session>(env, onnx_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;

    size_t n_in = session->GetInputCount();
    for (size_t i = 0; i < n_in; ++i) {
        auto name = session->GetInputNameAllocated(i, alloc);
        if (std::string(name.get()) == "attention_mask")
            needs_attention_mask = true;
    }

    size_t n_out = session->GetOutputCount();
    for (size_t i = 0; i < n_out; ++i) {
        auto name = session->GetOutputNameAllocated(i, alloc);
        if (std::string(name.get()) == "hidden_states") {
            has_hidden_states = true;
            break;
        }
    }
    loaded = true;
}

std::vector<float> Wav2Vec2Encoder::encode(const float* audio16k, int n_samples, int* T_out) {
    // wav2vec2 feature extractor requires zero-mean unit-variance input
    std::vector<float> norm(audio16k, audio16k + n_samples);
    float mean = 0.f;
    for (float x : norm) mean += x;
    mean /= (float)n_samples;
    float var = 0.f;
    for (float x : norm) var += (x - mean) * (x - mean);
    var /= (float)n_samples;
    float istd = 1.f / (std::sqrt(var) + 1e-7f);
    for (float& x : norm) x = (x - mean) * istd;

    std::vector<int64_t> in_shape = {1, n_samples};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_info, norm.data(), norm.size(),
        in_shape.data(), in_shape.size()));

    std::vector<float> attn_mask;
    if (needs_attention_mask) {
        attn_mask.assign(n_samples, 1.0f);
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, attn_mask.data(), attn_mask.size(),
            in_shape.data(), in_shape.size()));
    }

    const char* input_names_both[] = {"input_values", "attention_mask"};
    const char* output_names_hs[]  = {"hidden_states"};
    const char* output_names_lg[]  = {"logits"};

    int n_inputs      = needs_attention_mask ? 2 : 1;
    const char** out_names = has_hidden_states ? output_names_hs : output_names_lg;

    auto outputs = session->Run(
        Ort::RunOptions{nullptr},
        input_names_both, inputs.data(), n_inputs,
        out_names, 1);

    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t T = shape[1];
    int64_t D = shape[2];
    *T_out = static_cast<int>(T);

    float* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + T * D);
}
