#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 2-tap grain shifter — matches plugin exactly
class GrainShifter {
public:
    static constexpr uint32_t BUF  = 4096;
    static constexpr uint32_t MASK = BUF - 1;
    static constexpr double   GRAIN = 256.0;

    GrainShifter() : write_(0) {
        for (auto& s : buf_) s = 0.0f;
        phase_a_ = 0.0; phase_b_ = 0.5;
    }

    void reset_phases() {
        phase_a_ = 0.0; phase_b_ = 0.5;
    }

    float process(float in, double ratio) {
        buf_[write_ & MASK] = in;
        ++write_;
        const double inc = (1.0 - ratio) / GRAIN;

        phase_a_ += inc; phase_a_ -= std::floor(phase_a_);
        phase_b_ += inc; phase_b_ -= std::floor(phase_b_);

        float sa = lerp(phase_a_ * GRAIN + 2.0) * hann(phase_a_);
        float sb = lerp(phase_b_ * GRAIN + 2.0) * hann(phase_b_);
        return sa + sb;
    }

private:
    float    buf_[BUF];
    uint32_t write_;
    double   phase_a_, phase_b_;

    float lerp(double delay) const {
        double rp = (double)write_ - delay;
        rp = std::fmod(rp, (double)BUF);
        if (rp < 0.0) rp += BUF;
        uint32_t i0 = (uint32_t)rp & MASK;
        uint32_t i1 = (i0 + 1) & MASK;
        float frac  = (float)(rp - std::floor(rp));
        return buf_[i0] * (1.0f - frac) + buf_[i1] * frac;
    }

    static float hann(double p) {
        return 0.5f * (1.0f - (float)std::cos(2.0 * M_PI * p));
    }
};
