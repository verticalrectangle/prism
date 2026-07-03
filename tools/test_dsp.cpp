#include "dsp.h"
#include "output_ring.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

static int failures = 0;

static void pass(const char* name) { printf("[PASS] %s\n", name); }
static void fail(const char* name, const char* reason) {
    printf("[FAIL] %s — %s\n", name, reason);
    ++failures;
}
#define CHECK(name, cond, reason) do { if (cond) pass(name); else fail(name, reason); } while(0)

// ── 2a. OLA unity-gain invariant ─────────────────────────────────────────────
//
// Run the same OLA accumulation loop as ml_thread: Hann-window each chunk,
// overlap-add at 50% hop, emit kOlaHop samples per iteration.
// A DC input of 1.0 should come out as 1.0 after the first full window settles.

static void test_ola_unity_gain() {
    const char* name = "OLA unity-gain invariant";

    std::vector<float> hann(kOlaWin);
    compute_hann(hann.data(), kOlaWin);

    // Verify the window sums: hann[i] + hann[i + kOlaHop] == 1.0 for all i in [0, kOlaHop)
    float max_err = 0.f;
    for (int i = 0; i < kOlaHop; ++i) {
        float sum = hann[i] + hann[i + kOlaHop];
        float err = std::abs(sum - 1.f);
        if (err > max_err) max_err = err;
    }
    if (max_err > 1e-6f) {
        fail(name, "Hann windows at 50% overlap do not sum to 1.0");
        return;
    }

    // Simulate OLA with DC=1.0 signal for several hops
    std::vector<float> chunk(kOlaWin, 1.f);   // constant 1.0 — the "VITS output"
    std::vector<float> ola_buf(kOlaWin, 0.f);
    std::vector<float> output;

    for (int iter = 0; iter < 8; ++iter) {
        for (int i = 0; i < kOlaWin; ++i)
            ola_buf[i] += chunk[i] * hann[i];

        for (int i = 0; i < kOlaHop; ++i)
            output.push_back(ola_buf[i]);

        std::memmove(ola_buf.data(), ola_buf.data() + kOlaHop, kOlaHop * sizeof(float));
        std::memset(ola_buf.data() + kOlaHop, 0, kOlaHop * sizeof(float));
    }

    // Skip the first window (not yet fully settled), check the rest
    float out_max_err = 0.f;
    for (int i = kOlaWin; i < (int)output.size(); ++i) {
        float err = std::abs(output[i] - 1.f);
        if (err > out_max_err) out_max_err = err;
    }
    char reason[64];
    snprintf(reason, sizeof(reason), "max output error %.2e (tol 1e-5)", (double)out_max_err);
    CHECK(name, out_max_err < 1e-5f, reason);
}

// ── 2b. Resampler ─────────────────────────────────────────────────────────────

static void test_resampler_identity() {
    const char* name = "Resampler 1:1 identity";
    const int N = 1024;
    std::vector<float> in(N), out(N + 4);
    for (int i = 0; i < N; ++i) in[i] = std::sin(2.f * 3.14159265f * i / 32.f);

    size_t n_out = resample_linear(in.data(), N, out.data(), 44100, 44100);

    if ((int)n_out != N) { fail(name, "output length != input length"); return; }
    float max_err = 0.f;
    for (int i = 0; i < N; ++i) max_err = std::max(max_err, std::abs(out[i] - in[i]));
    char reason[64];
    snprintf(reason, sizeof(reason), "max error %.2e", (double)max_err);
    CHECK(name, max_err < 1e-5f, reason);
}

static void test_resampler_roundtrip_rms() {
    const char* name = "Resampler round-trip RMS (44100→16000→44100)";
    const int N = 44100;  // 1 second of 1 kHz sine @ 44.1kHz
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i) in[i] = std::sin(2.f * 3.14159265f * 1000.f * i / 44100.f);

    std::vector<float> down(16001), up(N + 16);
    size_t n_down = resample_linear(in.data(), N, down.data(), 44100, 16000);
    size_t n_up   = resample_linear(down.data(), n_down, up.data(), 16000, 44100);

    double rms_in = 0, rms_out = 0;
    size_t compare = std::min((size_t)N, n_up);
    for (size_t i = 0; i < compare; ++i) {
        rms_in  += in[i]   * in[i];
        rms_out += up[i]   * up[i];
    }
    rms_in  = std::sqrt(rms_in  / compare);
    rms_out = std::sqrt(rms_out / compare);

    // Allow up to 1 dB loss
    double db = 20.0 * std::log10(rms_out / (rms_in + 1e-12));
    char reason[64];
    snprintf(reason, sizeof(reason), "RMS loss %.2f dB (limit -1 dB)", db);
    CHECK(name, db > -1.0, reason);
}

// ── 2c. OutputRing ───────────────────────────────────────────────────────────

static void test_ring_pushpull() {
    const char* name = "OutputRing push/pull integrity";
    OutputRing ring;
    const int N = 1024;
    float src[N], dst[N] = {};
    for (int i = 0; i < N; ++i) src[i] = (float)i * 0.001f;

    ring.push(src, N);
    size_t got = ring.pull(dst, N);

    if (got != N) { fail(name, "pull returned wrong count"); return; }
    float max_err = 0.f;
    for (int i = 0; i < N; ++i) max_err = std::max(max_err, std::abs(dst[i] - src[i]));
    char reason[64];
    snprintf(reason, sizeof(reason), "max error %.2e", (double)max_err);
    CHECK(name, max_err < 1e-7f, reason);
}

static void test_ring_capacity() {
    const char* name = "OutputRing near-full capacity";
    OutputRing ring;
    const size_t N = OutputRing::CAP - 1;
    std::vector<float> src(N, 0.5f);

    ring.push(src.data(), N);
    size_t avail = ring.available();
    char reason[64];
    snprintf(reason, sizeof(reason), "available=%zu expected %zu", avail, N);
    CHECK(name, avail == N, reason);
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main() {
    test_ola_unity_gain();
    test_resampler_identity();
    test_resampler_roundtrip_rms();
    test_ring_pushpull();
    test_ring_capacity();

    printf("\n%s (%d failure%s)\n",
           failures == 0 ? "All tests passed." : "TESTS FAILED.",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
