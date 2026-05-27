#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include "prism.h"
#include "model.h"
#include "audio.h"
#include "ml_thread.h"
#include "lpc.h"
#include "hf_api.h"
#include "encoder_onnx.h"
#include "pth_reader.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

// ── WAV loader ────────────────────────────────────────────────────────────────

static std::vector<float> load_wav_mono(const std::string& path) {
    FILE* fp = popen(
        ("ffmpeg -hide_banner -loglevel error -i \"" + path +
         "\" -vn -ar 44100 -ac 1 -f f32le pipe:1 2>/dev/null").c_str(), "r");
    if (!fp) return {};
    std::vector<float> buf;
    float tmp[4096]; size_t n;
    while ((n = fread(tmp, sizeof(float), 4096, fp)) > 0)
        buf.insert(buf.end(), tmp, tmp + n);
    pclose(fp);
    return buf;
}

// ── GLFW ─────────────────────────────────────────────────────────────────────

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

// ── Inline export + load ──────────────────────────────────────────────────────

static void use_model(const std::string& pth_path, AppState& s, PrismModel& model) {
    ml_thread_stop(&s);
    s.model_loaded = false;

    std::string onnx = pth_path.substr(0, pth_path.rfind('.')) + ".onnx";

    if (fs::exists(onnx)) {
        try {
            model.load(onnx);
            s.model_loaded = true;
            snprintf(s.status_msg, sizeof(s.status_msg), "Model ready");
            if (s.target_loaded) ml_thread_start(&s, &model);
        } catch (const std::exception& e) {
            snprintf(s.status_msg, sizeof(s.status_msg), "Load error: %s", e.what());
        }
        return;
    }

    // Export needed — run on background thread
    s.export_status.store(1);
    snprintf(s.status_msg, sizeof(s.status_msg), "Exporting encoder...");

    std::thread([pth_path, onnx, &s, &model]() {
        PthModel pth = pth_open(pth_path);
        std::string err = pth.err.empty() ? conv_encoder_to_onnx(pth, onnx) : pth.err;
        pth_close(pth);
        if (!err.empty()) {
            s.export_error = err;
            s.export_status.store(3);
            return;
        }
        try {
            model.load(onnx);
            s.model_loaded = true;
        } catch (const std::exception& e) {
            s.export_error = e.what();
            s.export_status.store(3);
            return;
        }
        s.export_status.store(2);
    }).detach();
}

// ── UI ────────────────────────────────────────────────────────────────────────

static void draw_ui(AppState& s, PrismModel& model) {
    // Poll export thread status
    int exp = s.export_status.load();
    if (exp == 2) {
        snprintf(s.status_msg, sizeof(s.status_msg), "Model ready");
        if (s.target_loaded) ml_thread_start(&s, &model);
        s.export_status.store(0);
    } else if (exp == 3) {
        snprintf(s.status_msg, sizeof(s.status_msg), "Export failed: %s",
                 s.export_error.c_str());
        s.export_status.store(0);
    }

    int display_w, display_h;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &display_w, &display_h);

    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({(float)display_w, (float)display_h});
    ImGui::Begin("Prism", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    float w = ImGui::GetContentRegionAvail().x;

    // ── HuggingFace model browser ─────────────────────────────────────────────
    ImGui::Text("Voice Model");
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

    auto draw_card = [&](int id, const char* label, const char* repo, const char* file) {
        std::string key  = dl_key(repo, file);
        HFDownload& dl   = s_dl[key];
        hf_download_poll(dl);

        // Download done → auto-use
        if (dl.status.load(std::memory_order_acquire) == HFDownload::Status::Done) {
            use_model(dl.out_path, s, model);
            dl.status.store(HFDownload::Status::Idle, std::memory_order_release);
        }

        bool installed = hf_rvc_installed(repo, file);
        bool active    = (s.model_path == hf_rvc_model_path(repo, file));

        ImGui::PushID(id);
        ImVec2 cp = ImGui::GetCursorScreenPos();
        float cw = w, ch = 44.f;
        ImDrawList* dl_draw = ImGui::GetWindowDrawList();
        bool hov = ImGui::IsMouseHoveringRect(cp, {cp.x+cw, cp.y+ch});

        dl_draw->AddRectFilled(cp, {cp.x+cw, cp.y+ch},
            hov ? IM_COL32(20,36,32,255) : IM_COL32(14,24,20,255), 4.f);
        dl_draw->AddRect(cp, {cp.x+cw, cp.y+ch},
            installed ? IM_COL32(30,200,150,180) : IM_COL32(40,60,50,160), 4.f, 0, 1.f);
        dl_draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
                         {cp.x+8.f, cp.y+8.f}, IM_COL32(255,255,255,220), label);

        auto dst = dl.status.load(std::memory_order_acquire);
        if (dst == HFDownload::Status::Running) {
            float prog = dl.progress();
            float bx0 = cp.x+cw-94.f, bx1 = cp.x+cw-8.f, by = cp.y+18.f;
            dl_draw->AddRectFilled({bx0,by},{bx1,by+5.f},IM_COL32(20,60,45,255),2.f);
            dl_draw->AddRectFilled({bx0,by},{bx0+(bx1-bx0)*prog,by+5.f},
                                   IM_COL32(30,200,150,255),2.f);
            dl_draw->AddText({bx0, by+8.f}, IM_COL32(80,180,140,180), "Downloading...");
        } else if (installed) {
            dl_draw->AddText({cp.x+8.f, cp.y+28.f},
                active ? IM_COL32(30,220,150,255) : IM_COL32(60,140,100,180),
                active ? "Selected" : "Installed");
            ImGui::SetCursorScreenPos({cp.x+cw-44.f, cp.y+8.f});
            if (ImGui::SmallButton("Use##u")) {
                s.model_path = hf_rvc_model_path(repo, file);
                use_model(s.model_path, s, model);
            }
        } else {
            ImGui::SetCursorScreenPos({cp.x+cw-76.f, cp.y+ch/2.f-8.f});
            if (ImGui::SmallButton("Download##d"))
                hf_download_model(repo, file, hf_rvc_model_path(repo, file), dl);
            if (dst == HFDownload::Status::Error) {
                dl_draw->AddText({cp.x+8.f, cp.y+28.f}, IM_COL32(220,80,80,200), "Failed");
                if (ImGui::IsItemHovered() && !dl.error_msg.empty())
                    ImGui::SetTooltip("%s", dl.error_msg.c_str());
            }
        }

        ImGui::SetCursorScreenPos(cp);
        ImGui::InvisibleButton("##card", {cw-80.f, ch});
        ImGui::Dummy({0.f, ch + 3.f});
        ImGui::PopID();
    };

    auto ss = s_search.status.load(std::memory_order_acquire);
    if (!s_query[0]) {
        for (int i = 0; i < 4; ++i)
            draw_card(1000+i, k_pinned[i].label, k_pinned[i].repo, k_pinned[i].file);
    } else if (ss == HFSearch::Status::Running || s_debounce > 0.f) {
        ImGui::TextDisabled("Searching...");
    } else if (ss == HFSearch::Status::Error) {
        ImGui::TextColored({1.f,0.4f,0.4f,1.f}, "%s", s_search.error.c_str());
    } else if (ss == HFSearch::Status::Done) {
        if (s_search.results.empty()) {
            ImGui::TextDisabled("No models found.");
        }
        for (int i = 0; i < (int)s_search.results.size(); ++i) {
            const HFModel& m = s_search.results[i];
            std::string disp = m.repo;
            auto sl = disp.rfind('/');
            if (sl != std::string::npos) disp = disp.substr(sl+1);
            for (char& c : disp) if (c=='_'||c=='-') c=' ';
            draw_card(2000+i, disp.c_str(), m.repo.c_str(), m.model_file.c_str());
        }
    }

    ImGui::Separator();

    // ── Speaker WAV ──────────────────────────────────────────────────────────
    ImGui::Text("Target Speaker (.wav)");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-120.0f);
    ImGui::InputText("##speaker", s.speaker_path_buf, sizeof(s.speaker_path_buf));
    ImGui::SameLine();
    if (ImGui::Button("Load Speaker")) {
        s.speaker_path = s.speaker_path_buf;
        if (fs::exists(s.speaker_path)) {
            auto pcm = load_wav_mono(s.speaker_path);
            if (!pcm.empty()) {
                compute_speaker_lpc(pcm.data(), pcm.size(), 12, s.target_lpc);
                s.target_f0_mean = 150.0f;
                s.target_loaded  = true;
                snprintf(s.status_msg, sizeof(s.status_msg), "Speaker loaded: %s",
                    fs::path(s.speaker_path).filename().c_str());
                if (s.model_loaded && !s.ml_running.load())
                    ml_thread_start(&s, &model);
            } else {
                snprintf(s.status_msg, sizeof(s.status_msg), "Failed to decode: %s",
                    s.speaker_path_buf);
            }
        } else {
            snprintf(s.status_msg, sizeof(s.status_msg), "File not found: %s",
                s.speaker_path_buf);
        }
    }

    ImGui::Separator();

    // ── Parameters ───────────────────────────────────────────────────────────
    {
        float v = s.param_blend.load();
        if (ImGui::SliderFloat("Blend", &v, 0.0f, 1.0f, "%.2f"))
            s.param_blend.store(v);
        ImGui::SetItemTooltip("LPC warp intensity toward target speaker");
    }
    {
        float v = s.param_formant_shift.load();
        if (ImGui::SliderFloat("Formant Shift (st)", &v, -2.0f, 2.0f, "%.2f"))
            s.param_formant_shift.store(v);
        ImGui::SetItemTooltip("Shift resonant peaks up/down in semitones");
    }
    {
        float v = s.param_smoothing.load();
        if (ImGui::SliderFloat("Smoothing", &v, 0.0f, 1.0f, "%.2f"))
            s.param_smoothing.store(v);
        ImGui::SetItemTooltip("Coefficient chase rate (0=fast, 1=slow/stable)");
    }
    {
        float v = s.param_voiced_thresh.load();
        if (ImGui::SliderFloat("Voiced Threshold", &v, 0.0f, 1.0f, "%.2f"))
            s.param_voiced_thresh.store(v);
        ImGui::SetItemTooltip("YIN confidence cutoff — below = dry passthrough");
    }
    {
        float v = s.param_mix.load();
        if (ImGui::SliderFloat("Mix", &v, 0.0f, 1.0f, "%.2f"))
            s.param_mix.store(v);
        ImGui::SetItemTooltip("Wet/dry blend");
    }

    ImGui::Separator();

    // ── Status bar ────────────────────────────────────────────────────────────
    bool live = s.audio_active.load();
    bool ml   = s.ml_running.load();
    float ms  = s.ml_ms_per_chunk.load();

    ImGui::TextColored(live ? ImVec4{0.2f,1.f,0.2f,1.f} : ImVec4{0.5f,0.5f,0.5f,1.f},
                       live ? "● LIVE" : "○ IDLE");
    ImGui::SameLine();
    if (ml && ms > 0.f) ImGui::Text("ML: %.1fms/chunk", ms);
    else                ImGui::Text("ML: --");
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

    GLFWwindow* window = glfwCreateWindow(640, 480, "Prism", nullptr, nullptr);
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

    AppState   state;
    PrismModel model;
    ma_device  dev;

    if (!audio_init(&state, &dev)) {
        fprintf(stderr, "Failed to open audio device\n");
        return 1;
    }
    audio_start(&dev, &state);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        draw_ui(state, model);

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
