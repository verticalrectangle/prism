// Quick ONNX export tool for testing
#include "rvc_onnx.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.pth>\n", argv[0]); return 2; }
    std::string pth_path = argv[1];
    std::string onnx = pth_path.substr(0, pth_path.rfind('.')) + "_vits3.onnx";
    
    PthModel pth = pth_open(pth_path);
    if (!pth.err.empty()) { fprintf(stderr, "pth_open: %s\n", pth.err.c_str()); return 1; }
    
    std::string err = pth_to_onnx(pth, onnx);
    pth_close(pth);
    if (!err.empty()) { fprintf(stderr, "export: %s\n", err.c_str()); return 1; }
    
    printf("Exported: %s\n", onnx.c_str());
    return 0;
}
