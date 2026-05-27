#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

// ── HubertModel ───────────────────────────────────────────────────────────────
// Full HuBERT base (7 conv layers + 12 transformer layers).
// Input:  audio float32 [1, N]       (16 kHz mono)
// Output: features float32 [1, T, 768]
struct HubertModel {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "hubert"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo     mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    bool loaded = false;

    void load(const std::string& path);
    // Returns [T * 768] floats (row-major [T][768]). Sets *T_out to frame count.
    std::vector<float> infer(const float* pcm16k, int n, int* T_out);
};

// ── RvcModel ──────────────────────────────────────────────────────────────────
// RVC VITS decoder.
// Inputs:  phone [1,T,phone_dim], f0 [1,T], sid [1]
// Output:  audio [1,1,M] (at sr samples/sec)
struct RvcModel {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "rvc"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo     mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    bool loaded = false;
    int  phone_dim = 768;
    int  sr = 40000;

    void load(const std::string& path, int sr, int phone_dim);
    // Returns synthesized audio [M] samples at sr.
    std::vector<float> infer(const float* phone, int T, const float* f0, int sid = 0);
};

// Linear resampler — writes to out[], returns output sample count.
int resample_linear(const float* in, int n_in, float in_rate,
                    float* out, int out_cap, float out_rate);
