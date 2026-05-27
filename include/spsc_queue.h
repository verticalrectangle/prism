#pragma once

#include <atomic>
#include <cstddef>

struct CoeffPacket {
    float biquad[6][5];  // 6 stages × {b0,b1,b2,a1,a2}
    float pitch_ratio;
    float voiced;
};

template<size_t N>
struct SpscQueue {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");

    alignas(64) CoeffPacket buf[N];
    alignas(64) std::atomic<size_t> head{0};
    alignas(64) std::atomic<size_t> tail{0};

    bool push(const CoeffPacket& pkt) {
        size_t t = tail.load(std::memory_order_relaxed);
        size_t next = (t + 1) & (N - 1);
        if (next == head.load(std::memory_order_acquire))
            return false;
        buf[t] = pkt;
        tail.store(next, std::memory_order_release);
        return true;
    }

    bool pop(CoeffPacket& pkt) {
        size_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire))
            return false;
        pkt = buf[h];
        head.store((h + 1) & (N - 1), std::memory_order_release);
        return true;
    }
};
