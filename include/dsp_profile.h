#pragma once
#include "biquad.h"
#include <cstddef>

static constexpr int kPrimerChunksNeeded = 20;
static constexpr int kLpcOrder           = 12;

struct DspVoiceProfile {
    float lpc[kLpcOrder]      = {};
    float median_f0           = 0.f;
    float biquad_coeffs[6][5] = {};

    // Accumulator state — ml_thread only
    float lpc_accum[kLpcOrder] = {};
    float f0_buf[64]           = {};
    int   f0_count             = 0;
    int   frames               = 0;
};

// Called per voiced VITS output chunk from ml_thread
void dsp_profile_accumulate(DspVoiceProfile& p, const float* audio, int n, float f0);

// Called once at kPrimerChunksNeeded frames — averages and bakes biquad coeffs
void dsp_profile_finalize(DspVoiceProfile& p);

// Called from audio callback — applies LPC filter to mic input
void dsp_profile_apply(const DspVoiceProfile& p, const float* in, float* out,
                       int n, BiquadCascade& biquad);
