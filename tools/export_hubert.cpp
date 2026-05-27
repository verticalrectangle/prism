#include "pth_reader.h"
#include "hubert_onnx.h"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <hubert_base.pt> <out.onnx>\n", argv[0]);
        return 1;
    }
    fprintf(stderr, "Opening %s...\n", argv[1]);
    PthModel pth = pth_open(argv[1]);
    if (!pth.err.empty()) {
        fprintf(stderr, "Error: %s\n", pth.err.c_str());
        return 1;
    }
    fprintf(stderr, "Exporting full HuBERT to %s...\n", argv[2]);
    std::string err = hubert_to_onnx(pth, argv[2]);
    pth_close(pth);
    if (!err.empty()) {
        fprintf(stderr, "Export error: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "Done.\n");
    return 0;
}
