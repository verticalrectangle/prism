#include "model.h"
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

int resample_linear(const float* in, int n_in, float in_rate,
                    float* out, int out_cap, float out_rate)
{
    if (n_in <= 0 || out_cap <= 0) return 0;
    double ratio = (double)in_rate / out_rate;
    int n_out = std::min(out_cap, (int)(n_in / ratio));
    for (int i = 0; i < n_out; i++) {
        double src = i * ratio;
        int i0 = (int)src;
        int i1 = (i0 + 1 < n_in) ? i0 + 1 : i0;
        float frac = (float)(src - i0);
        out[i] = in[i0] * (1.f - frac) + in[i1] * frac;
    }
    return n_out;
}

void HubertModel::load(const std::string& path) {
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, path.c_str(), opts);
    loaded = true;
}

std::vector<float> HubertModel::infer(const float* pcm16k, int n, int* T_out) {
    std::vector<int64_t> shape = {1, (int64_t)n};
    // ONNX Runtime requires non-const pointer; data is read-only during Run
    auto inp = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(pcm16k), (size_t)n, shape.data(), 2);

    const char* in_names[]  = {"audio"};
    const char* out_names[] = {"features"};

    auto outs = session->Run(Ort::RunOptions{}, in_names, &inp, 1, out_names, 1);

    auto info = outs[0].GetTensorTypeAndShapeInfo();
    auto dims = info.GetShape();
    // Expected shape: [1, T, 768]
    int T = (dims.size() >= 3) ? (int)dims[1] : 1;
    if (T_out) *T_out = T;

    float* data = outs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + (size_t)T * 768);
}

void RvcModel::load(const std::string& path, int sr_, int phone_dim_) {
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, path.c_str(), opts);
    sr       = sr_;
    phone_dim = phone_dim_;
    loaded   = true;
}

std::vector<float> RvcModel::infer(const float* phone, int T, const float* f0, int sid_val) {
    std::vector<int64_t> ph_shape  = {1, (int64_t)T, (int64_t)phone_dim};
    std::vector<int64_t> f0_shape  = {1, (int64_t)T};
    std::vector<int64_t> sid_shape = {1};
    int64_t sid_data = (int64_t)sid_val;

    auto ph_t = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(phone), (size_t)T * phone_dim,
        ph_shape.data(), 3);
    auto f0_t = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(f0), (size_t)T,
        f0_shape.data(), 2);
    auto sid_t = Ort::Value::CreateTensor<int64_t>(
        mem, &sid_data, 1,
        sid_shape.data(), 1);

    Ort::Value inputs[3] = {std::move(ph_t), std::move(f0_t), std::move(sid_t)};
    const char* in_names[]  = {"phone", "f0", "sid"};
    const char* out_names[] = {"audio"};

    auto outs = session->Run(Ort::RunOptions{}, in_names, inputs, 3, out_names, 1);

    size_t M = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    float* data = outs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + M);
}
