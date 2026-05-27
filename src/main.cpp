#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include "prism.h"
#include "model.h"
#include "audio.h"
#include "ml_thread.h"
#include "hf_api.h"
#include "rvc_onnx.h"
#include "pth_reader.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>
#include <unistd.h>

namespace fs = std::filesystem;

// ── Binary directory (for bundled models/) ────────────────────────────────────

static std::string binary_dir() {
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, (size_t)sizeof(buf) - 1);
    if (n <= 0) return "./";
    buf[n] = '\0';
    std::string p(buf);
    auto last = p.rfind('/');
    return (last != std::string::npos) ? p.substr(0, last + 1) : "./";
}

static std::string find_hubert_onnx() {
    // Release: binary sits next to models/
    std::string bin = binary_dir();
    for (const char* rel : {"models/hubert.onnx", "../models/hubert.onnx"}) {
        std::string p = bin + rel;
        if (fs::exists(p)) return p;
    }
    // Development: run from repo root
    if (fs::exists("models/hubert.onnx")) return "models/hubert.onnx";
    return "";
}

// ── Sidecar JSON parser ───────────────────────────────────────────────────────

static int json_int_field(const char* json, const char* key) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char* p = strstr(json, needle);
    if (!p) return 0;
    p += strlen(needle);
    while (*p == ' ') ++p;
    return atoi(p);
}

// ── GLFW ──────────────────────────────────────────────────────────────────────

static void glfw_error_callback(int err, const char* desc) {
    fprintf(stderr, "GLFW error %d: %s\n", err, desc);
}

// ── HF browser data ───────────────────────────────────────────────────────────

struct PinnedVoice { const char* label; const char* repo; const char* file; };
static const PinnedVoice k_pinned[] = {
    { "Drake",         "binant/Drake_RVC",                                     "model.pth"        },
    { "Travis Scott",  "binant/Travis_Scott_-_RVC_-_1000_Epoch_48k",           "model.pth"        },
    { "The Weeknd",    "binant/The_Weeknd__RVC__1000_Epochs",                  "model.pth"        },
    { "Playboi Carti", "Shadow-AI/Playboi_Carti_Deep_Voice_300_Epochs_RVC_V2", "PBCDeepVoice.zip" },
};

static std::string dl_key(const std::string& repo, const std::string& file) {
    return repo + "::" + file;
}

// ── Model type detection ──────────────────────────────────────────────────────

static bool is_rvc_model(const PthModel& pth) {
    return pth.tensors.count("enc_p.emb_phone.weight") > 0;
}

// ── Inline RVC export + load ──────────────────────────────────────────────────

static void use_model(const std::string& pth_path, AppState& s,
                      HubertModel& hubert, RvcModel& rvc) {
    ml_thread_stop(&s);
    s.rvc_loaded.store(false);
    s.rvc_model_path = pth_path;

    std::string onnx = pth_path.substr(0, pth_path.rfind('.')) + ".onnx";
    std::string json = pth_path.substr(0, pth_path.rfind('.')) + ".json";

    auto load_rvc = [&](int sr, int pdim) {
        try {
            rvc.load(onnx, sr, pdim);
            s.rvc_sr       = sr;
            s.rvc_phone_dim = pdim;
            s.rvc_loaded.store(true);
            snprintf(s.status_msg, sizeof(s.status_msg), "Voice ready");
            if (s.hubert_loaded.load())
                ml_thread_start(&s, &hubert, &rvc);
        } catch (const std::exception& e) {
            snprintf(s.status_msg, sizeof(s.status_msg), "Load error: %s", e.what());
        }
    };

    if (fs::exists(onnx) && fs::exists(json)) {
        // Already exported — read sidecar and load
        FILE* jf = fopen(json.c_str(), "r");
        char jbuf[256] = {};
        if (jf) { fread(jbuf, 1, sizeof(jbuf) - 1, jf); fclose(jf); }
        int sr   = json_int_field(jbuf, "target_sr");
        int pdim = json_int_field(jbuf, "phone_dim");
        if (sr   <= 0) sr   = 40000;
        if (pdim <= 0) pdim = 768;
        load_rvc(sr, pdim);
        return;
    }

    // Export needed — run on background thread
    s.export_status.store(1);
    snprintf(s.status_msg, sizeof(s.status_msg), "Exporting voice model...");

    std::thread([pth_path, onnx, json, &s, &hubert, &rvc, load_rvc]() mutable {
        PthModel pth = pth_open(pth_path);
        if (!pth.err.empty()) {
            s.export_error = pth.err;
            s.export_status.store(3);
            return;
        }
        if (!is_rvc_model(pth)) {
            s.export_error = "Not an RVC model (missing enc_p.emb_phone.weight)";
            pth_close(pth);
            s.export_status.store(3);
            return;
        }
        std::string err = pth_to_onnx(pth, onnx);
        pth_close(pth);
        if (!err.empty()) {
            s.export_error = err;
            s.export_status.store(3);
            return;
        }
        // Read sidecar JSON written by pth_to_onnx
        FILE* jf = fopen(json.c_str(), "r");
        char jbuf[256] = {};
        if (jf) { fread(jbuf, 1, sizeof(jbuf) - 1, jf); fclose(jf); }
        int sr   = json_int_field(jbuf, "target_sr");
        int pdim = json_int_field(jbuf, "phone_dim");
        if (sr   <= 0) sr   = 40000;
        if (pdim <= 0) pdim = 768;
        s.export_status.store(2);
        load_rvc(sr, pdim);
    }).detach();
}

// ── UI ────────────────────────────────────────────────────────────────────────

static void draw_ui(AppState& s, HubertModel& hubert, RvcModel& rvc) {
    // Poll export thread
    int exp = s.export_status.load();
    if (exp == 3) {
        snprintf(s.status_msg, sizeof(s.status_msg), "Export failed: %s",
                 s.export_error.c_str());
        s.export_status.store(0);
    } else if (exp == 2) {
        s.export_status.store(0);
    }

    // Poll HuBERT init thread
    int hinit = s.hubert_init_status.load();
    if (hinit == 2) {
        snprintf(s.status_msg, sizeof(s.status_msg), "Ready");
        s.hubert_init_status.store(0);
    } else if (hinit == 3) {
        snprintf(s.status_msg, sizeof(s.status_msg), "HuBERT load error: %s",
                 s.hubert_init_error.c_str());
        s.hubert_init_status.store(0);
    }

    int display_w, display_h;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &display_w, &display_h);

    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({(float)display_w, (float)display_h});
    ImGui::Begin("Prism", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove    | ImGuiWindowFlags_NoCollapse);

    float w = ImGui::GetContentRegionAvail().x;

    // ── HuggingFace voice browser ─────────────────────────────────────────────
    ImGui::Text("Voice");
    ImGui::SetNextItemWidth(w);
    static HFSearch          s_search;
    static char              s_query[64] = {};
    static char              s_prev[64]  = {};
    static float             s_debounce  = 0.f;
    static std::map<std::string, HFDownload> s_dl;

    if (ImGui::InputTextWithHint("##hfq", "Search HuggingFace RVC voices…",
                                  s_query, sizeof(s_query))) {
        if (strcmp(s_query, s_prev) != 0) {
            memcpy(s_prev, s_query, sizeof(s_query));
            s_debounce = s_query[0] ? 0.8f : 0.f;
            if (!s_query[0]) hf_search_cancel(s_search);
        }
    }
    if (s_debounce > 0.f) {
        s_debounce -= ImGui::GetIO().DeltaTime;
        if (s_debounce <= 0.f) {
            hf_search_cancel(s_search);
            hf_search(s_query, s_search);
            s_debounce = 0.f;
        }
    }

    bool exporting = (s.export_status.load() == 1);

    auto draw_card = [&](int id, const char* label, const char* repo, const char* file) {
        std::string key = dl_key(repo, file);
        HFDownload& dl  = s_dl[key];
        hf_download_poll(dl);

        // Download done → auto-use
        if (dl.status.load(std::memory_order_acquire) == HFDownload::Status::Done) {
            use_model(dl.out_path, s, hubert, rvc);
            dl.status.store(HFDownload::Status::Idle, std::memory_order_release);
        }

        bool installed = hf_rvc_installed(repo, file);
        bool active    = (s.rvc_model_path == hf_rvc_model_path(repo, file));

        ImGui::PushID(id);
        ImVec2 cp = ImGui::GetCursorScreenPos();
        float  cw = w, ch = 44.f;
        ImDrawList* dl_draw = ImGui::GetWindowDrawList();
        bool hov = ImGui::IsMouseHoveringRect(cp, {cp.x + cw, cp.y + ch});

        dl_draw->AddRectFilled(cp, {cp.x + cw, cp.y + ch},
            hov ? IM_COL32(20, 36, 32, 255) : IM_COL32(14, 24, 20, 255), 4.f);
        dl_draw->AddRect(cp, {cp.x + cw, cp.y + ch},
            installed ? IM_COL32(30, 200, 150, 180) : IM_COL32(40, 60, 50, 160),
            4.f, 0, 1.f);
        dl_draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
                         {cp.x + 8.f, cp.y + 8.f}, IM_COL32(255, 255, 255, 220), label);

        auto dst = dl.status.load(std::memory_order_acquire);
        if (dst == HFDownload::Status::Running) {
            float prog = dl.progress();
            float bx0 = cp.x + cw - 94.f, bx1 = cp.x + cw - 8.f, by = cp.y + 18.f;
            dl_draw->AddRectFilled({bx0, by}, {bx1, by + 5.f}, IM_COL32(20, 60, 45, 255), 2.f);
            dl_draw->AddRectFilled({bx0, by}, {bx0 + (bx1 - bx0) * prog, by + 5.f},
                                   IM_COL32(30, 200, 150, 255), 2.f);
            dl_draw->AddText({bx0, by + 8.f}, IM_COL32(80, 180, 140, 180), "Downloading...");
        } else if (exporting && active) {
            dl_draw->AddText({cp.x + 8.f, cp.y + 28.f}, IM_COL32(220, 200, 80, 200), "Exporting...");
        } else if (installed) {
            dl_draw->AddText({cp.x + 8.f, cp.y + 28.f},
                active ? IM_COL32(30, 220, 150, 255) : IM_COL32(60, 140, 100, 180),
                active ? "Selected" : "Installed");
            ImGui::SetCursorScreenPos({cp.x + cw - 44.f, cp.y + 8.f});
            if (ImGui::SmallButton("Use##u") && !exporting) {
                s.rvc_model_path = hf_rvc_model_path(repo, file);
                use_model(s.rvc_model_path, s, hubert, rvc);
            }
        } else {
            ImGui::SetCursorScreenPos({cp.x + cw - 76.f, cp.y + ch / 2.f - 8.f});
            if (ImGui::SmallButton("Download##d") && !exporting)
                hf_download_model(repo, file, hf_rvc_model_path(repo, file), dl);
            if (dst == HFDownload::Status::Error) {
                dl_draw->AddText({cp.x + 8.f, cp.y + 28.f}, IM_COL32(220, 80, 80, 200), "Failed");
                if (ImGui::IsItemHovered() && !dl.error_msg.empty())
                    ImGui::SetTooltip("%s", dl.error_msg.c_str());
            }
        }

        ImGui::SetCursorScreenPos(cp);
        ImGui::InvisibleButton("##card", {cw - 80.f, ch});
        ImGui::Dummy({0.f, ch + 3.f});
        ImGui::PopID();
    };

    auto ss = s_search.status.load(std::memory_order_acquire);
    if (!s_query[0]) {
        for (int i = 0; i < 4; ++i)
            draw_card(1000 + i, k_pinned[i].label, k_pinned[i].repo, k_pinned[i].file);
    } else if (ss == HFSearch::Status::Running || s_debounce > 0.f) {
        ImGui::TextDisabled("Searching...");
    } else if (ss == HFSearch::Status::Error) {
        ImGui::TextColored({1.f, 0.4f, 0.4f, 1.f}, "%s", s_search.error.c_str());
    } else if (ss == HFSearch::Status::Done) {
        if (s_search.results.empty())
            ImGui::TextDisabled("No models found.");
        for (int i = 0; i < (int)s_search.results.size(); ++i) {
            const HFModel& m = s_search.results[i];
            std::string disp = m.repo;
            auto sl = disp.rfind('/');
            if (sl != std::string::npos) disp = disp.substr(sl + 1);
            for (char& c : disp) if (c == '_' || c == '-') c = ' ';
            draw_card(2000 + i, disp.c_str(), m.repo.c_str(), m.model_file.c_str());
        }
    }

    ImGui::Separator();

    // ── Parameters ───────────────────────────────────────────────────────────
    {
        float v = s.param_formant_shift.load();
        if (ImGui::SliderFloat("Pitch Shift (st)", &v, -12.0f, 12.0f, "%.1f"))
            s.param_formant_shift.store(v);
        ImGui::SetItemTooltip("Shift synthesized pitch up/down in semitones");
    }
    {
        float v = s.param_mix.load();
        if (ImGui::SliderFloat("Mix", &v, 0.0f, 1.0f, "%.2f"))
            s.param_mix.store(v);
        ImGui::SetItemTooltip("Wet/dry blend (0 = mic passthrough, 1 = full synthesis)");
    }

    ImGui::Separator();

    // ── Status bar ────────────────────────────────────────────────────────────
    bool live = s.audio_active.load();
    bool ml   = s.ml_running.load();
    float ms  = s.ml_ms_per_chunk.load();

    ImVec4 dot_col = s.rvc_loaded.load() && ml
                     ? ImVec4{0.2f, 1.f, 0.2f, 1.f}
                     : ImVec4{0.5f, 0.5f, 0.5f, 1.f};
    const char* dot_label = (s.rvc_loaded.load() && ml) ? "● LIVE" : "○ IDLE";
    (void)live;

    ImGui::TextColored(dot_col, "%s", dot_label);
    ImGui::SameLine();
    if (ml && ms > 0.f) ImGui::Text("ML: %.0fms/chunk", ms);
    else                ImGui::TextDisabled("ML: --");
    ImGui::SameLine();
    ImGui::TextUnformatted(s.status_msg);

    ImGui::End();
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(640, 460, "Prism", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    AppState    state;
    HubertModel hubert;
    RvcModel    rvc;
    ma_device   dev;

    if (!audio_init(&state, &dev)) {
        fprintf(stderr, "Failed to open audio device\n");
        return 1;
    }
    audio_start(&dev, &state);

    // Load bundled HuBERT in a background thread
    {
        std::string hub_path = find_hubert_onnx();
        if (hub_path.empty()) {
            snprintf(state.status_msg, sizeof(state.status_msg),
                     "models/hubert.onnx not found — use a release build");
            state.hubert_init_status.store(0);
        } else {
            state.hubert_init_status.store(1);
            std::thread([hub_path, &state, &hubert]() {
                try {
                    hubert.load(hub_path);
                    state.hubert_loaded.store(true);
                    state.hubert_init_status.store(2);
                } catch (const std::exception& e) {
                    state.hubert_init_error = e.what();
                    state.hubert_init_status.store(3);
                }
            }).detach();
        }
    }

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        draw_ui(state, hubert, rvc);

        ImGui::Render();
        int fw, fh;
        glfwGetFramebufferSize(window, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ml_thread_stop(&state);
    audio_stop(&dev, &state);
    ma_device_uninit(&dev);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
