#pragma once

#include <atomic>
#include <cstddef>

struct OutputRing {
    static constexpr size_t CAP = 65536;

    float buf[CAP];
    std::atomic<size_t> write_pos{0};
    std::atomic<size_t> read_pos{0};

    void push(const float* samples, size_t n) {
        size_t w = write_pos.load(std::memory_order_relaxed);
        for (size_t i = 0; i < n; ++i)
            buf[(w + i) & (CAP - 1)] = samples[i];
        write_pos.store((w + n) & (CAP - 1), std::memory_order_release);
    }

    size_t available() const {
        size_t w = write_pos.load(std::memory_order_acquire);
        size_t r = read_pos.load(std::memory_order_relaxed);
        return (w - r) & (CAP - 1);
    }

    // Returns number of samples actually read (may be less than n on underrun)
    size_t pull(float* out, size_t n) {
        size_t avail = available();
        size_t count = avail < n ? avail : n;
        size_t r = read_pos.load(std::memory_order_relaxed);
        for (size_t i = 0; i < count; ++i)
            out[i] = buf[(r + i) & (CAP - 1)];
        read_pos.store((r + count) & (CAP - 1), std::memory_order_release);
        return count;
    }

    void clear() {
        read_pos.store(write_pos.load(std::memory_order_acquire),
                       std::memory_order_release);
    }
};
