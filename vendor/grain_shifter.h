#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct GrainShifter {
    static constexpr uint32_t BUF_SIZE = 4096;
    static constexpr uint32_t MASK = BUF_SIZE - 1;

    float buf[BUF_SIZE] = {};
    uint32_t write_pos = 0;
    double phase_a = 0.0;
    double phase_b = 0.5;
    uint32_t grain_size = 256;

    void reset() {
        for (auto &s : buf) s = 0.0f;
        write_pos = 0;
        phase_a = 0.0;
        phase_b = 0.5;
    }

    float process(float in, double pitch_ratio) {
        buf[write_pos & MASK] = in;
        write_pos++;

        double phase_inc = (1.0 - pitch_ratio) / static_cast<double>(grain_size);
        phase_a += phase_inc;
        phase_b += phase_inc;
        phase_a -= std::floor(phase_a);
        phase_b -= std::floor(phase_b);

        double delay_a = phase_a * grain_size + 2.0;
        double delay_b = phase_b * grain_size + 2.0;

        auto lerp_read = [&](double delay) -> float {
            double read_pos = static_cast<double>(write_pos) - delay;
            double wrapped = std::fmod(read_pos, static_cast<double>(BUF_SIZE));
            if (wrapped < 0) wrapped += BUF_SIZE;
            uint32_t i0 = static_cast<uint32_t>(wrapped) & MASK;
            uint32_t i1 = (i0 + 1) & MASK;
            float frac = static_cast<float>(wrapped - std::floor(wrapped));
            return buf[i0] * (1.0f - frac) + buf[i1] * frac;
        };

        float gain_a = 0.5f * (1.0f - std::cos(2.0 * M_PI * phase_a));
        float gain_b = 0.5f * (1.0f - std::cos(2.0 * M_PI * phase_b));

        return lerp_read(delay_a) * gain_a + lerp_read(delay_b) * gain_b;
    }
};
