// rmvpe_onnx.cpp — RMVPE pitch tracking, streaming variant.
// Mirrors RVC's rmvpe.py front end:
//   MelSpectrogram(128 mels, sr 16000, win/n_fft 1024, hop 160, fmin 30,
//                  fmax 8000, slaney filterbank+norm, log(clamp(., 1e-5)))
//   mel frames padded to a multiple of 32 for the U-Net, cropped after.
//   decode: argmax + 9-bin weighted average of cents, threshold 0.03.
#include "rmvpe_onnx.h"

#include <onnxruntime_cxx_api.h>
#include <fftw3.h>
#include <cmath>
#include <cstring>
#include <memory>

static constexpr int kSR    = 16000;
static constexpr int kNFFT  = 1024;
static constexpr int kHop   = 160;
static constexpr int kMels  = 128;
static constexpr int kBins  = kNFFT / 2 + 1;   // 513
static constexpr float kFMin = 30.f;
static constexpr float kFMax = 8000.f;

// ── HTK mel scale (librosa htk=True), unnormalised triangles ─────────────────
// The rmvpe.onnx export expects HTK mels with norm=None. Verified empirically:
// slaney/slaney (pop-maker-studio's original front end) decodes a 220 Hz
// harmonic stack ~570 cents flat; htk/norm=None decodes it at 219.6 Hz.

static float hz_to_mel(float hz) {
    return 2595.f * std::log10(1.f + hz / 700.f);
}
static float mel_to_hz(float mel) {
    return 700.f * (std::pow(10.f, mel / 2595.f) - 1.f);
}

// Mel filterbank [kMels][kBins], plain triangles (no area normalisation).
static std::vector<float> mel_filterbank() {
    std::vector<float> fb((size_t)kMels * kBins, 0.f);
    float mel_lo = hz_to_mel(kFMin), mel_hi = hz_to_mel(kFMax);
    std::vector<float> pts(kMels + 2);
    for (int i = 0; i < kMels + 2; i++)
        pts[(size_t)i] = mel_to_hz(mel_lo + (mel_hi - mel_lo) * i / (kMels + 1));
    for (int m = 0; m < kMels; m++) {
        float f0 = pts[(size_t)m], f1 = pts[(size_t)m+1], f2 = pts[(size_t)m+2];
        for (int k = 0; k < kBins; k++) {
            float fk = (float)k * kSR / kNFFT;
            float up = (f1 > f0) ? (fk - f0) / (f1 - f0) : 0.f;
            float dn = (f2 > f1) ? (f2 - fk) / (f2 - f1) : 0.f;
            float w  = std::fmin(up, dn);
            if (w > 0.f) fb[(size_t)m * kBins + k] = w;
        }
    }
    return fb;
}

// ── Impl ─────────────────────────────────────────────────────────────────────

struct RmvpeImpl {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "rmvpe"};
    std::unique_ptr<Ort::Session> sess;
    std::string in_name, out_name;

    std::vector<float> fb = mel_filterbank();
    std::vector<float> win;                 // periodic Hann [kNFFT]
    std::vector<float> frame;               // FFT input [kNFFT]
    std::vector<fftwf_complex> spec;        // FFT output [kBins]
    fftwf_plan plan = nullptr;

    RmvpeImpl() : win(kNFFT), frame(kNFFT), spec(kBins) {
        for (int i = 0; i < kNFFT; i++)
            win[(size_t)i] = 0.5f * (1.f - std::cos(2.f * (float)M_PI * i / kNFFT));
        plan = fftwf_plan_dft_r2c_1d(kNFFT, frame.data(), spec.data(),
                                     FFTW_ESTIMATE);
    }
    ~RmvpeImpl() { if (plan) fftwf_destroy_plan(plan); }
};

RmvpeOnnx::RmvpeOnnx()  = default;
RmvpeOnnx::~RmvpeOnnx() { delete impl_; }

bool RmvpeOnnx::load(const std::string& path, int threads) {
    delete impl_;
    impl_ = nullptr;
    try {
        auto* im = new RmvpeImpl();
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // No spin-waiting: other sessions in this process need the cores.
        opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
        im->sess = std::make_unique<Ort::Session>(im->env, path.c_str(), opts);
        Ort::AllocatorWithDefaultOptions alloc;
        im->in_name  = im->sess->GetInputNameAllocated(0, alloc).get();
        im->out_name = im->sess->GetOutputNameAllocated(0, alloc).get();
        impl_ = im;
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<float> RmvpeOnnx::f0(const float* x, int n) {
    if (!impl_ || n <= 0) return {};
    RmvpeImpl& im = *impl_;

    // Reflect-pad by n_fft/2 each side (torch.stft center=True equivalent)
    int half = kNFFT / 2;
    std::vector<float> padded((size_t)n + kNFFT);
    for (int i = 0; i < half; i++)
        padded[(size_t)i] = x[(size_t)std::min(half - i, n - 1)];
    std::memcpy(padded.data() + half, x, (size_t)n * sizeof(float));
    for (int i = 0; i < half; i++)
        padded[(size_t)(half + n + i)] = x[(size_t)std::max(0, n - 2 - i)];

    int T = ((int)padded.size() - kNFFT) / kHop + 1;
    if (T <= 0) return {};

    // STFT magnitudes → log-mel, laid out [128, T_pad] with T padded to a
    // multiple of 32 for the U-Net.
    int T_pad = 32 * ((T - 1) / 32 + 1);
    std::vector<float> mel_in((size_t)kMels * T_pad, std::log(1e-5f));
    std::vector<float> mag(kBins);
    for (int t = 0; t < T; t++) {
        const float* src = padded.data() + (size_t)t * kHop;
        for (int i = 0; i < kNFFT; i++)
            im.frame[(size_t)i] = src[i] * im.win[(size_t)i];
        fftwf_execute(im.plan);
        for (int k = 0; k < kBins; k++)
            mag[(size_t)k] = std::sqrt(im.spec[(size_t)k][0] * im.spec[(size_t)k][0] +
                                       im.spec[(size_t)k][1] * im.spec[(size_t)k][1]);
        for (int m = 0; m < kMels; m++) {
            const float* w = im.fb.data() + (size_t)m * kBins;
            double acc = 0.0;
            for (int k = 0; k < kBins; k++) acc += (double)w[k] * mag[(size_t)k];
            mel_in[(size_t)m * T_pad + t] = std::log(std::fmax((float)acc, 1e-5f));
        }
    }

    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                         OrtMemTypeDefault);
        std::vector<int64_t> shape = {1, kMels, T_pad};
        Ort::Value in = Ort::Value::CreateTensor<float>(
            mem, mel_in.data(), mel_in.size(), shape.data(), 3);

        const char* ins[]  = {im.in_name.c_str()};
        const char* outs[] = {im.out_name.c_str()};
        auto out = im.sess->Run(Ort::RunOptions{nullptr}, ins, &in, 1, outs, 1);

        auto oshape = out[0].GetTensorTypeAndShapeInfo().GetShape(); // [1,Tp,360]
        const float* sal = out[0].GetTensorData<float>();
        int n_bins = (int)oshape[2];   // 360

        // Decode: argmax + 9-bin weighted cents average
        std::vector<float> f0v((size_t)T, 0.f);
        for (int t = 0; t < T; t++) {
            const float* row = sal + (size_t)t * n_bins;
            int   best = 0;
            float maxv = row[0];
            for (int b = 1; b < n_bins; b++)
                if (row[b] > maxv) { maxv = row[b]; best = b; }
            if (maxv <= 0.03f) continue;
            double ws = 0.0, cs = 0.0;
            for (int d = -4; d <= 4; d++) {
                int b = best + d;
                if (b < 0 || b >= n_bins) continue;
                float cents = 20.f * b + 1997.3794084376191f;
                ws += (double)row[b];
                cs += (double)row[b] * cents;
            }
            if (ws > 0.0)
                f0v[(size_t)t] = 10.f * std::pow(2.f, (float)(cs / ws) / 1200.f);
        }
        return f0v;
    } catch (...) {
        return {};
    }
}
