#include "hubert_encoder.h"

#include <stdexcept>

void HubertEncoder::load(const std::string& onnx_path, int n_threads) {
    opts.SetIntraOpNumThreads(n_threads > 0 ? n_threads : 2);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
    session = std::make_unique<Ort::Session>(env, onnx_path.c_str(), opts);
    loaded = true;
}

std::vector<float> HubertEncoder::encode(const float* audio16k, int n_samples, int* T_out) {
    // HuBERT takes raw 16kHz audio — no normalization needed
    std::vector<int64_t> in_shape = {1, n_samples};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(audio16k), (size_t)n_samples,
        in_shape.data(), in_shape.size()));

    const char* input_names[]  = {"audio"};
    const char* output_names[] = {"features"};

    auto outputs = session->Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), 1,
        output_names, 1);

    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t T = shape[1];
    int64_t D = shape[2];
    *T_out = static_cast<int>(T);

    float* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + T * D);
}
