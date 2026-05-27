#pragma once
// encoder_onnx.h — Export the HuBERT convolutional feature extractor to ONNX.
// Only the 7 conv layers are included (no transformer, no feature projection).
// Input:  audio float32 [1, N]   (16 kHz mono)
// Output: features float32 [1, 512, T]
// Returns "" on success, error string on failure.
#include "pth_reader.h"
#include <string>

std::string conv_encoder_to_onnx(const PthModel& m, const std::string& out_path);
