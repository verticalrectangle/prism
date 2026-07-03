// test_rmvpe — sanity + latency check for the streaming RMVPE port.
// Feeds a 400ms window of synthetic tones at prism's ML hop size and checks
// the decoded f0 against ground truth, then reports per-call latency.
#include "rmvpe_onnx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static constexpr int kSR  = 16000;
static constexpr int kWin = 6400;   // 400ms, prism's ML window

int main() {
    const char* home = getenv("HOME");
    std::string path = std::string(home ? home : "") + "/.cache/prism/models/rmvpe.onnx";

    RmvpeOnnx rmvpe;
    if (!rmvpe.load(path, 4)) {
        fprintf(stderr, "FAIL: cannot load %s\n", path.c_str());
        return 1;
    }

    int fails = 0;
    for (float hz : {110.f, 220.f, 440.f, 523.25f, 880.f}) {
        // Harmonic stack (voice-like): pure sines are out-of-distribution for
        // RMVPE and decode poorly; real voices carry many harmonics.
        std::vector<float> x(kWin, 0.f);
        for (int h = 1; h <= 12 && hz * h < 7000.f; h++)
            for (int i = 0; i < kWin; i++)
                x[(size_t)i] += (0.5f / h) * std::sin(2.f * (float)M_PI * hz * h * i / kSR);
        auto f0 = rmvpe.f0(x.data(), kWin);
        if (f0.empty()) { printf("%7.1f Hz: FAIL (empty)\n", hz); fails++; continue; }
        // Median over the central frames (edges see the reflect padding)
        std::vector<float> mid(f0.begin() + 8, f0.end() - 8);
        std::sort(mid.begin(), mid.end());
        float med = mid[mid.size() / 2];
        float err_cents = 1200.f * std::log2(med / hz);
        bool ok = std::fabs(err_cents) < 50.f;
        printf("%7.1f Hz: decoded %7.1f Hz (%+.0f cents) %s\n",
               hz, med, err_cents, ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }

    // Silence must be unvoiced
    {
        std::vector<float> x(kWin, 0.f);
        auto f0 = rmvpe.f0(x.data(), kWin);
        int voiced = 0;
        for (float v : f0) if (v > 0.f) voiced++;
        printf("silence: %d/%zu voiced frames %s\n", voiced, f0.size(),
               voiced == 0 ? "ok" : "FAIL");
        if (voiced != 0) fails++;
    }

    // Latency: steady-state per-call cost at the ML hop rate
    {
        std::vector<float> x(kWin);
        for (int i = 0; i < kWin; i++)
            x[(size_t)i] = 0.5f * std::sin(2.f * (float)M_PI * 220.f * i / kSR);
        rmvpe.f0(x.data(), kWin);  // warm-up
        auto t0 = std::chrono::steady_clock::now();
        constexpr int kIters = 20;
        for (int i = 0; i < kIters; i++) rmvpe.f0(x.data(), kWin);
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count() / kIters;
        printf("latency: %.1f ms per 400ms window (budget 100ms/hop)\n", ms);
        if (ms > 90.f) { printf("latency: FAIL — too slow for real time\n"); fails++; }
    }

    printf(fails ? "FAILED (%d)\n" : "ALL OK\n", fails);
    return fails ? 1 : 0;
}
