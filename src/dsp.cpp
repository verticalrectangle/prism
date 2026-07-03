#include "dsp.h"

#include <cmath>
#include <algorithm>

static constexpr int kYINHop = 160;
static constexpr int kYINWin = 1024;

size_t resample_linear(const float* in, size_t n_in, float* out,
                       int in_rate, int out_rate) {
    size_t n_out = (size_t)((double)n_in * out_rate / in_rate);
    double ratio = (double)n_in / (double)n_out;
    for (size_t i = 0; i < n_out; ++i) {
        double src  = i * ratio;
        size_t i0   = (size_t)src;
        size_t i1   = (i0 + 1 < n_in) ? i0 + 1 : i0;
        float  frac = (float)(src - i0);
        out[i] = in[i0] * (1.f - frac) + in[i1] * frac;
    }
    return n_out;
}

std::vector<float> yin_f0(const float* x, int N, int sr) {
    int half     = kYINWin / 2;
    int min_tau  = std::max(1, sr / 1100);
    int max_tau  = std::min(half - 1, sr / 50);
    int n_frames = std::max(1, (N - kYINWin) / kYINHop + 1);
    std::vector<float> f0(n_frames, 0.f), d(half);

    for (int i = 0; i < n_frames; i++) {
        int start = i * kYINHop;
        for (int tau = 0; tau < half; tau++) {
            float sum = 0.f;
            for (int j = 0; j < half; j++) {
                float a = (start + j       < N) ? x[start + j]       : 0.f;
                float b = (start + j + tau < N) ? x[start + j + tau] : 0.f;
                sum += (a - b) * (a - b);
            }
            d[tau] = sum;
        }
        d[0] = 1.f;
        float run = 0.f;
        for (int tau = 1; tau < half; tau++) {
            run += d[tau];
            d[tau] = (run > 1e-10f) ? d[tau] * tau / run : 1.f;
        }
        for (int tau = std::max(min_tau, 2); tau < max_tau; tau++) {
            if (d[tau] < 0.15f && d[tau] <= d[tau - 1]) {
                float denom = d[tau-1] - 2.f*d[tau] + (tau+1 < half ? d[tau+1] : d[tau]);
                float adj   = (std::abs(denom) > 1e-10f && tau+1 < half)
                            ? 0.5f * (d[tau-1] - d[tau+1]) / denom : 0.f;
                adj = std::max(-0.5f, std::min(0.5f, adj));
                f0[i] = (float)sr / (tau + adj);
                break;
            }
        }
    }
    return f0;
}

std::vector<float> interp_f0(const std::vector<float>& src, int target) {
    if ((int)src.size() == target) return src;
    std::vector<float> out(target);
    float scale = (float)((int)src.size() - 1) / std::max(target - 1, 1);
    for (int i = 0; i < target; i++) {
        float pos = i * scale;
        int   j   = (int)pos;
        float t   = pos - j;
        int   k   = std::min(j + 1, (int)src.size() - 1);
        out[i]    = src[j] * (1.f - t) + src[k] * t;
    }
    return out;
}

void compute_hann(float* out, int len) {
    for (int i = 0; i < len; ++i)
        out[i] = 0.5f * (1.f - std::cos(2.f * 3.14159265f * i / len));
}
