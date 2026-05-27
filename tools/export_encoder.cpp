// export_encoder.cpp — CLI: hubert_base.pt → prism_encoder.onnx (conv layers only)
//
// Exports only the 7-layer convolutional feature extractor from a fairseq
// HuBERT base checkpoint. The transformer layers are NOT included — this is
// what keeps inference to ~3ms per 40ms chunk on CPU.
//
// Input:  audio float32 [1, N]   (16 kHz mono, raw PCM)
// Output: features float32 [1, 512, T]
//
// Usage:
//   ./prism-export-encoder hubert_base.pt prism_encoder.onnx
//
#include "pth_reader.h"
#include "encoder_onnx.h"
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char* argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <hubert_base.pt> <output.onnx>\n", argv[0]);
        return 1;
    }

    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];

    fprintf(stdout, "Opening %s ...\n", in_path.c_str());
    fflush(stdout);

    PthModel m = pth_open(in_path);
    if (!m.err.empty()) {
        fprintf(stderr, "Error opening .pth: %s\n", m.err.c_str());
        return 1;
    }

    fprintf(stdout, "Loaded %zu tensors.\n", m.tensors.size());

    if (m.tensors.empty()) {
        fprintf(stderr, "ERROR: no tensors found in checkpoint.\n");
        pth_close(m);
        return 1;
    }

    {
        std::vector<std::string> names;
        names.reserve(m.tensors.size());
        for (auto& kv : m.tensors) names.push_back(kv.first);
        std::sort(names.begin(), names.end());
        size_t show = std::min(names.size(), (size_t)20);
        fprintf(stdout, "First %zu tensor names:\n", show);
        for (size_t i = 0; i < show; i++)
            fprintf(stdout, "  %s\n", names[i].c_str());
        if (names.size() > show)
            fprintf(stdout, "  ... (%zu more)\n", names.size() - show);
        fflush(stdout);
    }

    fprintf(stdout, "Exporting conv encoder to %s ...\n", out_path.c_str());
    fflush(stdout);

    std::string err = conv_encoder_to_onnx(m, out_path);
    pth_close(m);

    if (!err.empty()) {
        fprintf(stderr, "Export failed: %s\n", err.c_str());
        return 1;
    }

    fprintf(stdout, "Success: wrote %s\n", out_path.c_str());
    return 0;
}
