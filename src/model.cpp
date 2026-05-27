#include "model.h"

#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

// Linear decimation from 44100 to 16000 (ratio ≈ 2.75625)
static size_t resample_44k_to_16k(const float* in, size_t n_in, float* out) {
    constexpr double ratio = static_cast<double>(kInRate) / kHubRate;
    size_t n_out = static_cast<size_t>(n_in / ratio);
    for (size_t i = 0; i < n_out; ++i) {
        double src = i * ratio;
        size_t i0 = static_cast<size_t>(src);
        size_t i1 = i0 + 1 < n_in ? i0 + 1 : i0;
        float frac = static_cast<float>(src - i0);
        out[i] = in[i0] * (1.0f - frac) + in[i1] * frac;
    }
    return n_out;
}

void PrismModel::load(const std::string& onnx_path) {
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, onnx_path.c_str(), session_options);
    loaded = true;
}

bool PrismModel::is_loaded() const { return loaded; }

// Infer HuBERT conv features from 44.1 kHz samples.
// Returns pointer to internal feat_buf (kFeatDim floats), valid until next call.
const float* PrismModel::infer(const float* samples_44k, size_t n) {
    // Resample to 16 kHz
    size_t n_16k = resample_44k_to_16k(samples_44k, n, resample_buf.data());

    // Run ONNX session
    // Input shape: [1, n_16k]
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(n_16k)};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, resample_buf.data(), n_16k, input_shape.data(), input_shape.size());

    const char* input_names[]  = {"input"};
    const char* output_names[] = {"output"};

    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    // Output shape: [1, T, 512] or [1, 512, T] — take mean over time
    float* out_data = output_tensors[0].GetTensorMutableData<float>();
    auto   out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto   out_shape = out_info.GetShape();

    int64_t T = 1, feat_dim = kFeatDim;
    if (out_shape.size() == 3) {
        // Determine layout: [1, T, 512] vs [1, 512, T]
        if (out_shape[2] == kFeatDim) {
            T = out_shape[1]; feat_dim = kFeatDim;
        } else {
            feat_dim = out_shape[1]; T = out_shape[2];
        }
    }

    // Mean-pool over time axis
    memset(feat_buf.data(), 0, sizeof(float) * kFeatDim);
    if (out_shape.size() == 3 && out_shape[2] == kFeatDim) {
        // Layout [1, T, 512]
        for (int64_t t = 0; t < T; ++t)
            for (int64_t f = 0; f < feat_dim; ++f)
                feat_buf[f] += out_data[t * feat_dim + f];
    } else {
        // Layout [1, 512, T]
        for (int64_t f = 0; f < feat_dim; ++f)
            for (int64_t t = 0; t < T; ++t)
                feat_buf[f] += out_data[f * T + t];
    }
    float inv_T = T > 0 ? 1.0f / T : 1.0f;
    for (int f = 0; f < kFeatDim; ++f)
        feat_buf[f] *= inv_T;

    return feat_buf.data();
}
