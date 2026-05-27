#pragma once

#include <cstring>
#include <algorithm>

struct BiquadStage {
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    float tb0 = 1, tb1 = 0, tb2 = 0, ta1 = 0, ta2 = 0;
    float z1 = 0, z2 = 0;
};

struct BiquadCascade {
    static constexpr int STAGES = 6;

    BiquadStage stages[STAGES];
    int ramp_remain = 0;

    void set_coeffs(const float c[6][5], int ramp_samples) {
        for (int s = 0; s < STAGES; ++s) {
            stages[s].tb0 = c[s][0];
            stages[s].tb1 = c[s][1];
            stages[s].tb2 = c[s][2];
            stages[s].ta1 = c[s][3];
            stages[s].ta2 = c[s][4];
        }
        ramp_remain = ramp_samples > 0 ? ramp_samples : 1;
    }

    float process(float x) {
        if (ramp_remain > 0) {
            float alpha = 1.0f / static_cast<float>(ramp_remain);
            for (int s = 0; s < STAGES; ++s) {
                stages[s].b0 += (stages[s].tb0 - stages[s].b0) * alpha;
                stages[s].b1 += (stages[s].tb1 - stages[s].b1) * alpha;
                stages[s].b2 += (stages[s].tb2 - stages[s].b2) * alpha;
                stages[s].a1 += (stages[s].ta1 - stages[s].a1) * alpha;
                stages[s].a2 += (stages[s].ta2 - stages[s].a2) * alpha;
            }
            --ramp_remain;
        }

        for (int s = 0; s < STAGES; ++s) {
            auto& st = stages[s];
            float y = st.b0 * x + st.z1;
            st.z1 = st.b1 * x - st.a1 * y + st.z2;
            st.z2 = st.b2 * x - st.a2 * y;
            x = y;
        }
        return x;
    }

    void reset() {
        for (auto& s : stages) { s.z1 = 0; s.z2 = 0; }
    }
};
