#pragma once
// RMVPE neural pitch tracker (ONNX) — streaming port of pop-maker-studio's
// rmvpe_onnx. Log-mel front end (FFTW) + U-Net salience model + local-average
// cents decode: argmax + 9-bin weighted cents average, voicing threshold 0.03.
// Far more robust than YIN: no octave jumps, clean voiced/unvoiced decisions.
//
// Unlike the offline original, this holds the ONNX session, FFTW plan, and mel
// filterbank as instance state so it can be called once per hop from the ML
// thread without per-call setup cost. Single-threaded use only (no locking).
#include <string>
#include <vector>

struct RmvpeImpl;

class RmvpeOnnx {
public:
    RmvpeOnnx();
    ~RmvpeOnnx();
    RmvpeOnnx(const RmvpeOnnx&) = delete;
    RmvpeOnnx& operator=(const RmvpeOnnx&) = delete;

    // Load rmvpe.onnx (input log-mel [1,128,T], output salience [1,T,360]).
    // Returns false if the file is missing or the session fails to build.
    bool load(const std::string& path, int threads);
    bool loaded() const { return impl_ != nullptr; }

    // Estimate F0 from 16 kHz mono audio. Returns one value per 10 ms frame
    // (0.0 = unvoiced). Empty on error or if not loaded.
    std::vector<float> f0(const float* x, int n);

private:
    RmvpeImpl* impl_ = nullptr;
};
