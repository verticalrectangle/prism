#include "lpc.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <complex>
#include <vector>

void levinson_durbin(const float* autocorr, int order, float* lpc_out) {
    std::vector<float> a(order, 0.0f);
    std::vector<float> tmp(order, 0.0f);

    float err = autocorr[0];
    if (err < 1e-10f) {
        memset(lpc_out, 0, sizeof(float) * order);
        return;
    }

    for (int i = 0; i < order; ++i) {
        float lambda = 0.0f;
        for (int j = 0; j < i; ++j)
            lambda += a[j] * autocorr[i - j];
        lambda = (autocorr[i + 1] - lambda) / err;

        for (int j = 0; j < i; ++j)
            tmp[j] = a[j] - lambda * a[i - 1 - j];
        tmp[i] = lambda;
        for (int j = 0; j <= i; ++j)
            a[j] = tmp[j];

        err *= (1.0f - lambda * lambda);
        if (err < 1e-10f) break;
    }

    memcpy(lpc_out, a.data(), sizeof(float) * order);
}

void features_to_autocorr(const float* features, int feat_dim,
                           float* autocorr, int order) {
    // Project feature vector onto order+1 autocorrelation lags via dot products
    // with deterministic projection vectors (hash-based to avoid storing a matrix)
    for (int lag = 0; lag <= order; ++lag) {
        float sum = 0.0f;
        for (int k = 0; k < feat_dim; ++k) {
            // Simple deterministic projection: cos of index-based phase
            float proj = std::cos(static_cast<float>(lag * k + lag) * 0.0314159f);
            sum += features[k] * proj;
        }
        // Bias the autocorrelation toward positive-definite by squaring lag-0
        autocorr[lag] = (lag == 0) ? sum * sum + 1.0f : sum;
    }
    // Ensure R[0] dominates (positive-definite constraint)
    float r0 = autocorr[0];
    if (r0 < 1.0f) {
        float scale = 1.0f / (r0 + 1e-6f);
        for (int lag = 0; lag <= order; ++lag)
            autocorr[lag] *= scale;
        autocorr[0] = 1.0f;
    }
}

// Find roots of LPC polynomial (companion matrix eigenvalues via QR iteration)
// Simplified: pair complex roots into biquad sections, order must be even
static void find_roots(const float* lpc, int order,
                       std::vector<std::complex<float>>& roots) {
    // Bairstow's method for paired root extraction
    roots.clear();
    std::vector<float> poly(order + 1);
    poly[0] = 1.0f;
    for (int i = 0; i < order; ++i)
        poly[i + 1] = lpc[i];

    int remaining = order;
    std::vector<float> p = poly;

    while (remaining >= 2) {
        // Initial guess for quadratic factor x^2 + ux + v
        float u = -0.5f, v = 0.1f;

        for (int iter = 0; iter < 200; ++iter) {
            int n = remaining;
            std::vector<float> b(n + 1), c(n + 1);
            b[0] = p[0];
            b[1] = (n >= 1) ? p[1] - u * b[0] : 0.0f;
            for (int i = 2; i <= n; ++i)
                b[i] = p[i] - u * b[i - 1] - v * b[i - 2];

            c[0] = b[0];
            c[1] = (n >= 1) ? b[1] - u * c[0] : 0.0f;
            for (int i = 2; i <= n; ++i)
                c[i] = b[i] - u * c[i - 1] - v * c[i - 2];

            float det = c[n - 2] * c[n - 2] - c[n - 3] * (c[n - 1] - b[n]);
            if (std::fabs(det) < 1e-12f) { u += 0.01f; v += 0.01f; continue; }

            float du = (-b[n - 1] * c[n - 2] + b[n] * c[n - 3]) / det;
            float dv = (-b[n] * c[n - 2] + b[n - 1] * (c[n - 1] - b[n])) / det;
            u += du; v += dv;

            if (std::fabs(du) + std::fabs(dv) < 1e-7f) break;
        }

        // Extract complex root pair from x^2 + ux + v
        float disc = u * u - 4.0f * v;
        if (disc < 0) {
            float re = -u * 0.5f;
            float im = std::sqrt(-disc) * 0.5f;
            roots.emplace_back(re, im);
            roots.emplace_back(re, -im);
        } else {
            float r1 = (-u + std::sqrt(disc)) * 0.5f;
            float r2 = (-u - std::sqrt(disc)) * 0.5f;
            roots.emplace_back(r1, 0.0f);
            roots.emplace_back(r2, 0.0f);
        }

        // Deflate polynomial by x^2 + ux + v
        int n = remaining;
        std::vector<float> newp(n - 1);
        newp[0] = p[0];
        if (n >= 2) newp[1] = p[1] - u * p[0];
        for (int i = 2; i < n - 1; ++i)
            newp[i] = p[i] - u * newp[i - 1] - v * newp[i - 2];
        p = newp;
        remaining -= 2;
    }
}

void lpc_to_biquad(const float* lpc, int order, float out[6][5], float gain) {
    // Convert LPC poles to biquad cascade (order must be ≤ 12, exactly 6 stages)
    std::vector<std::complex<float>> roots;
    find_roots(lpc, order, roots);

    // Pair complex conjugates into biquads, sort by frequency for stability
    int n_stages = std::min((int)roots.size() / 2, 6);

    for (int s = 0; s < 6; ++s) {
        out[s][0] = 1.0f; // b0
        out[s][1] = 0.0f; // b1
        out[s][2] = 0.0f; // b2
        out[s][3] = 0.0f; // a1
        out[s][4] = 0.0f; // a2
    }

    for (int s = 0; s < n_stages; ++s) {
        auto r = roots[s * 2];  // complex root (its conjugate is roots[s*2+1])
        float re = r.real();
        float im = r.imag();

        // Clamp pole magnitude to 0.99 for stability
        float mag = std::sqrt(re * re + im * im);
        if (mag > 0.99f) {
            re *= 0.99f / mag;
            im *= 0.99f / mag;
        }

        // H(z) = 1 / (1 - 2*re*z^-1 + (re^2+im^2)*z^-2)
        out[s][0] = (s == 0) ? gain : 1.0f;
        out[s][1] = 0.0f;
        out[s][2] = 0.0f;
        out[s][3] = -2.0f * re;
        out[s][4] = re * re + im * im;
    }
}

void compute_speaker_lpc(const float* pcm, size_t n, int order, float* lpc_out) {
    // Compute autocorrelation of the entire PCM buffer
    std::vector<float> autocorr(order + 1, 0.0f);
    for (int lag = 0; lag <= order; ++lag) {
        float sum = 0.0f;
        for (size_t i = 0; i < n - lag; ++i)
            sum += pcm[i] * pcm[i + lag];
        autocorr[lag] = sum;
    }
    levinson_durbin(autocorr.data(), order, lpc_out);
}
