# Prism

Real-time CPU-only voice conversion. No training required. Single binary.

Point it at a pre-trained HuBERT conv encoder (`.onnx`) and a target speaker recording (`.wav`), and your mic input shifts toward that speaker's voice character in real time with less than 20ms perceived latency.

---

## Architecture

Three threads, all lock-free:

```
Mic input
    │
    ▼
Audio thread (miniaudio duplex callback)
    ├── InputRing ──────────────────────► ML thread
    │                                         │
    │   ◄─── SpscQueue (CoeffPacket) ─────────┘
    │
    ▼
GrainShifter → BiquadCascade → output
```

**Audio thread** — runs in the miniaudio callback, no malloc, no locks:
- Writes raw mic samples to `InputRing` for the ML thread
- Pops the latest `CoeffPacket` from the `SpscQueue` (produced by ML thread)
- Applies `GrainShifter` (2-tap Hann-windowed grain pitch shift) then `BiquadCascade` (6 second-order sections, ramped to prevent clicks)
- Wet/dry mix to speakers

**ML thread** — 40ms processing loop:
- Pulls 1764 samples (40ms @ 44.1 kHz) from `InputRing`
- Resamples to 16 kHz, runs HuBERT conv ONNX inference → 512-dim feature vector
- `features_to_autocorr` → `levinson_durbin` → 12th-order LPC coefficients
- Lerps source LPC toward target speaker LPC (Blend parameter)
- Applies semitone-based formant shift by scaling LPC coefficients
- `lpc_to_biquad` → 6 second-order sections via Bairstow root finding
- YIN pitch detection → `pitch_ratio = clamp(target_f0 / src_f0, 0.5, 2.0)`
- Pushes `CoeffPacket` to `SpscQueue`

**UI thread** — ImGui, GLFW + OpenGL 3.3:
- Model file picker (.onnx)
- Target speaker picker (.wav) — LPC computed once at load
- Parameter sliders
- Live status (ML ms/chunk, active indicator)

---

## Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| Blend | 0–1 | LPC warp intensity toward target speaker voice character |
| Formant Shift | -2 to +2 st | Slide resonant peaks up or down |
| Smoothing | 0–1 | Coefficient chase rate (0 = fast/jittery, 1 = slow/stable) |
| Voiced Threshold | 0–1 | YIN confidence cutoff — below this, dry passthrough |
| Mix | 0–1 | Wet/dry blend |

---

## Dependencies

- GLFW 3
- OpenGL 3.3
- ONNX Runtime ≥ 1.20
- A C++17 compiler, CMake ≥ 3.20, Ninja

No FFmpeg, no FFTW, no aubio, no Python at runtime.

---

## Build

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/prism
```

ONNX Runtime headers and libs are expected at `/usr/include/onnxruntime` and `/usr/lib`. To install:

```bash
curl -sL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz -o /tmp/ort.tgz
tar -xzf /tmp/ort.tgz -C /tmp/
sudo cp -r /tmp/onnxruntime-linux-x64-1.20.1/include/* /usr/include/onnxruntime/
sudo cp -r /tmp/onnxruntime-linux-x64-1.20.1/lib/* /usr/lib/
sudo ldconfig
```

---

## Model setup

Prism uses only the 7-layer convolutional feature extractor from HuBERT base (input `[1, N]` → output `[1, 512, T]`). The 12 transformer layers are skipped — the conv stack alone runs in ~3ms per 40ms chunk on CPU.

Export from a fairseq HuBERT base `.pt` checkpoint using the included tool:

```bash
./build/prism-export-encoder hubert_base.pt prism_encoder.onnx
```

The exporter uses the same hand-rolled pickle VM and protobuf writer as [Pop Maker Studio](https://github.com/verticalrectangle/pop-maker-studio) — no Python, no PyTorch, no protobuf library required. Weight-norm is resolved at export time. Run once; the `.onnx` is cached.

---

## Usage

1. Launch `./build/prism`
2. Enter the path to your exported `prism_encoder.onnx` and click **Load Model**
3. Enter the path to a target speaker `.wav` (any length, mono or stereo) and click **Load Speaker**
4. Speak into your mic — the status bar shows `● LIVE` and ML latency in ms/chunk
5. Adjust Blend to taste; Formant Shift and Voiced Threshold shape character further

---

## Related

- [Pop Maker Studio](https://github.com/verticalrectangle/pop-maker-studio) — video/lyric editor that shares the pth_reader pickle VM and ONNX inference pattern
