#pragma once

#include <cstddef>

// Levinson-Durbin recursion: autocorr[0..order] → lpc_out[0..order-1]
void levinson_durbin(const float* autocorr, int order, float* lpc_out);

// Map HuBERT conv feature vector to LPC autocorrelation via dot products
void features_to_autocorr(const float* features, int feat_dim,
                           float* autocorr, int order);

// Convert LPC coefficients to 6-stage biquad cascade (paired poles)
// gain scales b0 of the first stage
void lpc_to_biquad(const float* lpc, int order, float out[6][5], float gain);

// Compute target speaker LPC from raw PCM (call once at load)
void compute_speaker_lpc(const float* pcm, size_t n, int order, float* lpc_out);
