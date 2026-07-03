#pragma once

#include <cstddef>
#include <vector>

// ── Sample-rate conversion ────────────────────────────────────────────────────

size_t resample_linear(const float* in, size_t n_in, float* out,
                       int in_rate, int out_rate);

// ── F0 estimation (YIN) ───────────────────────────────────────────────────────

std::vector<float> yin_f0(const float* x, int N, int sr);
std::vector<float> interp_f0(const std::vector<float>& src, int target);

// ── Overlap-add constants and window ─────────────────────────────────────────

static constexpr int kOlaHop = 4800;   // 100ms @ 48kHz
static constexpr int kOlaWin = 9600;   // 200ms @ 48kHz — 50% overlap

// Fill out[0..len) with a Hann window: 0.5*(1 - cos(2π*i/len))
void compute_hann(float* out, int len);
