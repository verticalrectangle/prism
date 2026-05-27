#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include "prism.h"
#include "model.h"
#include "audio.h"
#include "ml_thread.h"
#include "lpc.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

// ── WAV loader (PCM float, mono) ─────────────────────────────────────────────

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

// ── GLFW callbacks ───────────────────────────────────────────────────────────

static void glfw_error_callback(int err, const char* desc) {
    fprintf(stderr, "GLFW error %d: %s\n", err, desc);
}

// ── UI ───────────────────────────────────────────────────────────────────────

static void draw_ui(AppState& s, PrismModel& model, ma_device& dev) {
    int display_w, display_h;
    GLFWwindow* win = glfwGetCurrentContext();
    glfwGetFramebufferSize(win, &display_w, &display_h);

    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({(float)display_w, (float)display_h});
    ImGui::Begin("Prism", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    // ── Model file ───────────────────────────────────────────────────────────
    ImGui::Text("Model (.pth or .onnx)");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-120.0f);
    ImGui::InputText("##model", s.model_path_buf, sizeof(s.model_path_buf));
    ImGui::SameLine();
    if (ImGui::Button("Load Model")) {
        s.model_path = s.model_path_buf;
        if (fs::exists(s.model_path)) {
            try {
                ml_thread_stop(&s);
                model.load(s.model_path);
                s.model_loaded = true;
                snprintf(s.status_msg, sizeof(s.status_msg), "Model loaded: %s",
                    fs::path(s.model_path).filename().c_str());
                if (s.target_loaded)
                    ml_thread_start(&s, &model);
            } catch (const std::exception& e) {
                snprintf(s.status_msg, sizeof(s.status_msg), "Model error: %s", e.what());
                s.model_loaded = false;
            }
        } else {
            snprintf(s.status_msg, sizeof(s.status_msg), "File not found: %s", s.model_path_buf);
        }
    }

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
                // Compute mean F0 estimate (simple energy-based fallback: 150 Hz default)
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
        float blend = s.param_blend.load();
        if (ImGui::SliderFloat("Blend", &blend, 0.0f, 1.0f, "%.2f"))
            s.param_blend.store(blend);
        ImGui::SetItemTooltip("LPC warp intensity toward target speaker");
    }
    {
        float fs_ = s.param_formant_shift.load();
        if (ImGui::SliderFloat("Formant Shift (st)", &fs_, -2.0f, 2.0f, "%.2f"))
            s.param_formant_shift.store(fs_);
        ImGui::SetItemTooltip("Shift resonant peaks up/down in semitones");
    }
    {
        float sm = s.param_smoothing.load();
        if (ImGui::SliderFloat("Smoothing", &sm, 0.0f, 1.0f, "%.2f"))
            s.param_smoothing.store(sm);
        ImGui::SetItemTooltip("Coefficient chase rate (0=fast, 1=slow/stable)");
    }
    {
        float vt = s.param_voiced_thresh.load();
        if (ImGui::SliderFloat("Voiced Threshold", &vt, 0.0f, 1.0f, "%.2f"))
            s.param_voiced_thresh.store(vt);
        ImGui::SetItemTooltip("YIN confidence cutoff — below this = dry passthrough");
    }
    {
        float mix = s.param_mix.load();
        if (ImGui::SliderFloat("Mix", &mix, 0.0f, 1.0f, "%.2f"))
            s.param_mix.store(mix);
        ImGui::SetItemTooltip("Wet/dry blend");
    }

    ImGui::Separator();

    // ── Status bar ───────────────────────────────────────────────────────────
    bool live = s.audio_active.load();
    bool ml   = s.ml_running.load();
    float ms  = s.ml_ms_per_chunk.load();

    if (live) {
        ImGui::TextColored({0.2f, 1.0f, 0.2f, 1.0f}, "● LIVE");
    } else {
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "○ IDLE");
    }
    ImGui::SameLine();

    if (ml && ms > 0.0f)
        ImGui::Text("ML: %.1fms/chunk", ms);
    else
        ImGui::Text("ML: --");

    ImGui::SameLine();
    ImGui::TextUnformatted(s.status_msg);

    ImGui::End();
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(640, 340, "Prism", nullptr, nullptr);
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
    PrismModel  model;
    ma_device   dev;

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

        draw_ui(state, model, dev);

        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
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
