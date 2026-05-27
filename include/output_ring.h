#pragma once
#include <atomic>
#include <cstddef>

// Lock-free SPSC ring buffer for synthesized audio (ML thread → audio thread).
struct OutputRing {
    static constexpr size_t CAP = 131072;  // 2^17, ~3s at 44.1kHz

    float             buf[CAP];
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

    // Pulls up to n samples. Returns count actually read.
    size_t pull(float* out, size_t n) {
        size_t avail = available();
        if (avail < n) n = avail;
        size_t r = read_pos.load(std::memory_order_relaxed);
        for (size_t i = 0; i < n; ++i)
            out[i] = buf[(r + i) & (CAP - 1)];
        read_pos.store((r + n) & (CAP - 1), std::memory_order_release);
        return n;
    }
};
