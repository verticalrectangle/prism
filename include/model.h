#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

static constexpr int kFeatDim  = 512;
static constexpr int kHubRate  = 16000;
static constexpr int kInRate   = 44100;
// Max 44.1k samples per inference chunk (1764 = 40ms @ 44.1k)
static constexpr int kChunk44k = 1764;
// Resampled size: ceil(1764 * 16000 / 44100) = 641
static constexpr int kChunk16k = 641;

struct PrismModel {
    Ort::Env     env{ORT_LOGGING_LEVEL_WARNING, "prism"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> resample_buf = std::vector<float>(kChunk16k + 4, 0.0f);
    std::vector<float> feat_buf     = std::vector<float>(kFeatDim, 0.0f);
    bool loaded = false;

    void load(const std::string& onnx_path);
    bool is_loaded() const;

    // Infer HuBERT conv features from 44.1 kHz mono samples.
    // Returns internal buffer (kFeatDim floats), valid until next call.
    // NOT thread-safe — call only from ML thread.
    const float* infer(const float* samples_44k, size_t n);
};
