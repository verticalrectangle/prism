#include "dsp_profile.h"
#include "lpc.h"

#include <algorithm>
#include <cmath>
#include <cstring>

void dsp_profile_accumulate(DspVoiceProfile& p, const float* audio, int n, float f0) {
    float lpc[kLpcOrder];
    compute_speaker_lpc(audio, (size_t)n, kLpcOrder, lpc);
    for (int i = 0; i < kLpcOrder; ++i)
        p.lpc_accum[i] += lpc[i];

    if (p.f0_count < 64)
        p.f0_buf[p.f0_count++] = f0;

    ++p.frames;
}

void dsp_profile_finalize(DspVoiceProfile& p) {
    if (p.frames < 1) return;

    for (int i = 0; i < kLpcOrder; ++i)
        p.lpc[i] = p.lpc_accum[i] / (float)p.frames;

    int n = p.f0_count;
    float sorted[64];
    std::copy(p.f0_buf, p.f0_buf + n, sorted);
    std::sort(sorted, sorted + n);
    p.median_f0 = sorted[n / 2];

    lpc_to_biquad(p.lpc, kLpcOrder, p.biquad_coeffs, 0.1f);
}

void dsp_profile_apply(const DspVoiceProfile& p, const float* in, float* out,
                       int n, BiquadCascade& biquad) {
    for (int i = 0; i < n; ++i)
        out[i] = biquad.process(in[i]);
}
