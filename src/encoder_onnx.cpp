// encoder_onnx.cpp — Export HuBERT convolutional feature extractor to ONNX.
// Only the 7 conv layers — no transformer, no feature projection.
// Input:  audio float32 [1, N]   (16 kHz mono)
// Output: features float32 [1, 512, T]
//
// Adapted from pop-maker-studio/src/hubert_onnx.cpp (same protobuf/graph infra,
// same emit_feature_extractor — just a different graph I/O and no transformer).
#include "encoder_onnx.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

// ── Protobuf writer ───────────────────────────────────────────────────────────

struct PbBuf {
    std::vector<uint8_t> d;
    void push_bytes(const void* p, size_t n) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
        d.insert(d.end(), b, b + n);
    }
    void varint(uint64_t v) {
        do { uint8_t b = v & 0x7F; v >>= 7; if (v) b |= 0x80; d.push_back(b); } while (v);
    }
    void tag_varint(int field, uint64_t v) { varint(((uint64_t)field<<3)|0); varint(v); }
    void tag_i32(int field, uint32_t v)    { varint(((uint64_t)field<<3)|5); push_bytes(&v,4); }
    void tag_bytes(int field, const void* p, size_t n) {
        varint(((uint64_t)field<<3)|2); varint(n); push_bytes(p,n);
    }
    void tag_string(int field, const std::string& s) { tag_bytes(field, s.data(), s.size()); }
    void tag_msg(int field, const PbBuf& sub) { tag_bytes(field, sub.d.data(), sub.d.size()); }
    void append(const PbBuf& o) { d.insert(d.end(), o.d.begin(), o.d.end()); }
};

static constexpr int kDtFloat = 1, kDtInt64 = 7;
static constexpr int kAttrFloat = 1, kAttrInt = 2, kAttrInts = 7;

static PbBuf make_tensor_float(const std::string& name, const std::vector<int64_t>& dims,
                                const std::vector<float>& data) {
    PbBuf t;
    for (int64_t d : dims) t.tag_varint(1, (uint64_t)d);
    t.tag_varint(2, kDtFloat); t.tag_string(8, name);
    t.tag_bytes(9, data.data(), data.size() * sizeof(float));
    return t;
}
static PbBuf make_tensor_int64(const std::string& name, const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& data) {
    PbBuf t;
    for (int64_t d : dims) t.tag_varint(1, (uint64_t)d);
    t.tag_varint(2, kDtInt64); t.tag_string(8, name);
    t.tag_bytes(9, data.data(), data.size() * sizeof(int64_t));
    return t;
}

// ── Graph builder ─────────────────────────────────────────────────────────────

struct OnnxGraph {
    PbBuf nodes, initializers, inputs, outputs;
    int ctr = 0;
    std::string uid(const std::string& p) { return p + "_" + std::to_string(ctr++); }

    void add_init_f32(const std::string& n, const std::vector<int64_t>& d, const std::vector<float>& v)
        { initializers.tag_msg(5, make_tensor_float(n,d,v)); }
    void add_init_i64(const std::string& n, const std::vector<int64_t>& d, const std::vector<int64_t>& v)
        { initializers.tag_msg(5, make_tensor_int64(n,d,v)); }
    void add_scalar_i64(const std::string& n, int64_t v) { add_init_i64(n,{},{v}); }
    void add_scalar_f32(const std::string& n, float v)   { add_init_f32(n,{},{v}); }

    void add_node(const std::string& op, const std::vector<std::string>& ins,
                  const std::vector<std::string>& outs, const std::string& nm,
                  const std::vector<PbBuf>& attrs = {}) {
        PbBuf n;
        for (auto& s : ins)  n.tag_string(1, s);
        for (auto& s : outs) n.tag_string(2, s);
        n.tag_string(3, nm); n.tag_string(4, op);
        for (auto& a : attrs) n.tag_msg(5, a);
        nodes.tag_msg(1, n);
    }
    std::string emit(const std::string& op, const std::vector<std::string>& ins,
                     const std::vector<PbBuf>& attrs = {}, const std::string& hint = "") {
        std::string out = uid(hint.empty() ? op : hint);
        add_node(op, ins, {out}, out+"_n", attrs);
        return out;
    }

    static PbBuf attr_int(const std::string& n, int64_t v) {
        PbBuf a; a.tag_string(1,n); a.tag_varint(3,(uint64_t)v); a.tag_varint(20,kAttrInt); return a;
    }
    static PbBuf attr_float(const std::string& n, float v) {
        PbBuf a; a.tag_string(1,n); uint32_t b; memcpy(&b,&v,4); a.tag_i32(4,b);
        a.tag_varint(20,kAttrFloat); return a;
    }
    static PbBuf attr_ints(const std::string& n, const std::vector<int64_t>& v) {
        PbBuf a; a.tag_string(1,n); PbBuf p;
        for (int64_t x : v) p.varint((uint64_t)x);
        a.tag_bytes(8,p.d.data(),p.d.size()); a.tag_varint(20,kAttrInts); return a;
    }

    static PbBuf make_value_info(const std::string& name, int dtype,
                                  const std::vector<int64_t>& dims,
                                  const std::vector<std::string>& dyn = {}) {
        PbBuf shape; int dc = 0;
        for (size_t i = 0; i < dims.size(); i++) {
            PbBuf dim;
            if (dims[i] < 0) dim.tag_string(2, dc < (int)dyn.size() ? dyn[dc++] : "dyn"+std::to_string(dc++));
            else              dim.tag_varint(1, (uint64_t)dims[i]);
            shape.tag_msg(1, dim);
        }
        PbBuf tt; tt.tag_varint(1,(uint64_t)dtype); tt.tag_msg(2,shape);
        PbBuf tp; tp.tag_msg(1,tt);
        PbBuf vi; vi.tag_string(1,name); vi.tag_msg(2,tp); return vi;
    }
    void add_input(const std::string& n, int dt, const std::vector<int64_t>& d,
                   const std::vector<std::string>& dn = {})
        { inputs.tag_msg(11, make_value_info(n,dt,d,dn)); }
    void add_output(const std::string& n, int dt, const std::vector<int64_t>& d,
                    const std::vector<std::string>& dn = {})
        { outputs.tag_msg(12, make_value_info(n,dt,d,dn)); }

    std::vector<uint8_t> serialise(const std::string& gname) const {
        PbBuf graph; graph.append(nodes); graph.tag_string(2,gname);
        graph.append(initializers); graph.append(inputs); graph.append(outputs);
        PbBuf opset; opset.tag_string(1,""); opset.tag_varint(2,17);
        PbBuf model; model.tag_varint(1,8); model.tag_msg(8,opset); model.tag_msg(7,graph);
        return model.d;
    }
};

// ── Weight-norm ───────────────────────────────────────────────────────────────

static std::vector<float> apply_weight_norm(const std::vector<float>& wv,
                                             const std::vector<float>& wg,
                                             const std::vector<int64_t>& shape) {
    int64_t dim0 = shape[0], rest = 1;
    for (size_t i = 1; i < shape.size(); i++) rest *= shape[i];
    std::vector<float> w(wv.size());
    for (int64_t i = 0; i < dim0; i++) {
        float gv = wg[(size_t)i];
        const float* src = wv.data() + (size_t)i*rest;
        double n2 = 0;
        for (int64_t k = 0; k < rest; k++) n2 += (double)src[k]*src[k];
        float sc = n2 > 1e-12 ? gv/(float)std::sqrt(n2) : 0.f;
        float* dst = w.data() + (size_t)i*rest;
        for (int64_t k = 0; k < rest; k++) dst[k] = src[k]*sc;
    }
    return w;
}

// ── Tensor loader ─────────────────────────────────────────────────────────────

struct TensorLoader {
    const PthModel& m;
    std::vector<float> load(const std::string& name) const {
        auto v = pth_load_tensor(m, name);
        if (v.empty()) throw std::runtime_error("missing tensor: " + name);
        return v;
    }
    std::vector<float> load_wn(const std::string& base) const {
        auto iv = m.tensors.find(base+".weight_v");
        auto ig = m.tensors.find(base+".weight_g");
        if (iv != m.tensors.end() && ig != m.tensors.end()) {
            auto wv = load(base+".weight_v"), wg = load(base+".weight_g");
            return apply_weight_norm(wv, wg, iv->second.shape);
        }
        return load(base+".weight");
    }
    bool has(const std::string& n) const { return m.tensors.count(n) > 0; }
};

// ── Op helpers ────────────────────────────────────────────────────────────────

using G = OnnxGraph;

static std::string op_add(G& g, const std::string& a, const std::string& b, const std::string& h="add")
    { return g.emit("Add",{a,b},{},h); }
static std::string op_mul(G& g, const std::string& a, const std::string& b, const std::string& h="mul")
    { return g.emit("Mul",{a,b},{},h); }
static std::string op_div(G& g, const std::string& a, const std::string& b, const std::string& h="div")
    { return g.emit("Div",{a,b},{},h); }

static std::string op_conv1d(G& g, const std::string& x, const std::string& w, const std::string& b,
                               int kernel, int stride, int padding, const std::string& hint="conv") {
    std::vector<PbBuf> attrs = {
        G::attr_ints("dilations",{1}), G::attr_ints("kernel_shape",{kernel}),
        G::attr_ints("pads",{padding,padding}), G::attr_ints("strides",{stride}),
        G::attr_int("group",1)
    };
    std::vector<std::string> ins = {x,w};
    if (!b.empty()) ins.push_back(b);
    return g.emit("Conv",ins,attrs,hint);
}

static std::string op_gelu(G& g, const std::string& x, const std::string& hint) {
    std::string sq2 = hint+"_sq2"; g.add_scalar_f32(sq2, std::sqrt(2.f));
    std::string one = hint+"_1f";  g.add_scalar_f32(one, 1.f);
    std::string hlf = hint+"_hf";  g.add_scalar_f32(hlf, 0.5f);
    auto xd  = op_div(g, x, sq2, hint+"_div");
    auto erf = g.emit("Erf",{xd},{},hint+"_erf");
    auto ep1 = op_add(g, erf, one, hint+"_ep1");
    auto mux = op_mul(g, x, ep1, hint+"_mux");
    return op_mul(g, mux, hlf, hint+"_gelu");
}

// ── Feature extractor: 7 conv layers → [1, 512, T] ──────────────────────────

static const int kFEStrides[7] = {5, 2, 2, 2, 2, 2, 2};
static const int kFEKernels[7] = {10, 3, 3, 3, 3, 2, 2};

static std::string emit_feature_extractor(G& g, const TensorLoader& tl, const std::string& x) {
    std::string cur = x;
    for (int i = 0; i < 7; i++) {
        std::string h    = "fe" + std::to_string(i);
        std::string cpfx = "feature_extractor.conv_layers." + std::to_string(i) + ".0";
        std::string npfx = "feature_extractor.conv_layers." + std::to_string(i) + ".2";
        int64_t c_out = 512, c_in = (i == 0) ? 1 : 512;
        int k = kFEKernels[i], s = kFEStrides[i];

        auto conv_w = tl.load(cpfx + ".weight");
        g.add_init_f32(h+"_cw", {c_out,c_in,k}, conv_w);
        cur = op_conv1d(g, cur, h+"_cw", "", k, s, 0, h+"_conv");

        if (i == 0) {
            auto gn_w = tl.load(npfx+".weight"), gn_b = tl.load(npfx+".bias");
            g.add_init_f32(h+"_gnw",{c_out},gn_w); g.add_init_f32(h+"_gnb",{c_out},gn_b);
            PbBuf eps; eps.tag_string(1,"epsilon");
            uint32_t eb; float ev = 1e-5f; memcpy(&eb,&ev,4);
            eps.tag_i32(4,eb); eps.tag_varint(20,kAttrFloat);
            cur = g.emit("InstanceNormalization",{cur,h+"_gnw",h+"_gnb"},{eps},h+"_gn");
        }
        cur = op_gelu(g, cur, h+"_gelu");
    }
    return cur;  // [1, 512, T]
}

// ── Entry point ───────────────────────────────────────────────────────────────

std::string conv_encoder_to_onnx(const PthModel& m, const std::string& out_path) {
    if (!m.err.empty()) return "PthModel error: " + m.err;
    try {
        TensorLoader tl{m};
        OnnxGraph g;

        g.add_input("input",  kDtFloat, {1,-1}, {"N"});
        g.add_output("output", kDtFloat, {1,512,-1}, {"T"});

        // [1, N] → [1, 1, N] → feature extractor → [1, 512, T]
        std::string unsq_ax = "au_ax"; g.add_init_i64(unsq_ax, {1}, {1});
        auto audio_3d = g.emit("Unsqueeze", {"input", unsq_ax}, {}, "au3d");
        auto features = emit_feature_extractor(g, tl, audio_3d);
        g.add_node("Identity", {features}, {"output"}, "out_id");

        auto bytes = g.serialise("prism_conv_encoder");
        std::ofstream ofs(out_path, std::ios::binary);
        if (!ofs) return "Cannot open for writing: " + out_path;
        ofs.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
        if (!ofs) return "Write error: " + out_path;
        return "";
    } catch (const std::exception& e) {
        return std::string("conv_encoder_to_onnx error: ") + e.what();
    }
}
