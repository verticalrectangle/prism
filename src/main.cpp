#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include "prism.h"
#include "audio.h"
#include "miniaudio.h"
#include "ml_thread.h"
#include "hf_api.h"
#include "pth_reader.h"
#include "rvc_onnx.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

// ── GLFW ─────────────────────────────────────────────────────────────────────

static void glfw_error_callback(int err, const char* desc) {
    fprintf(stderr, "GLFW error %d: %s\n", err, desc);
}

// ── HF browser data ───────────────────────────────────────────────────────────

struct PinnedVoice { const char* label; const char* repo; const char* file; };
static const PinnedVoice k_pinned[] = {
    { "Hatsune Miku",  "binant/Hatsune_Miku__RVC_v2_",                         "model.pth"        },
    { "Drake",         "binant/Drake_RVC",                                     "model.pth"        },
    { "Travis Scott",  "binant/Travis_Scott_-_RVC_-_1000_Epoch_48k",           "model.pth"        },
    { "The Weeknd",    "binant/The_Weeknd__RVC__1000_Epochs",                  "model.pth"        },
    { "Playboi Carti", "Shadow-AI/Playboi_Carti_Deep_Voice_300_Epochs_RVC_V2", "PBCDeepVoice.zip" },
};

static std::string dl_key(const std::string& repo, const std::string& file) {
    return repo + "::" + file;
}

// ── Read sidecar JSON from pth_to_onnx ───────────────────────────────────────

static void read_vits_json(const std::string& onnx_path, AppState& s) {
    std::string jp = onnx_path.substr(0, onnx_path.rfind('.')) + ".json";
    std::ifstream jf(jp);
    if (!jf) return;
    std::string js((std::istreambuf_iterator<char>(jf)),
                   std::istreambuf_iterator<char>());

    int target_sr = 40000, phone_dim = 768;
    auto read_int = [&](const char* key, int& out) {
        auto pos = js.find(key);
        if (pos != std::string::npos) out = atoi(js.c_str() + pos + strlen(key));
    };
    read_int("\"target_sr\":", target_sr);
    read_int("\"phone_dim\":", phone_dim);
    s.output_sr.store(target_sr);
    s.rvc_version.store(phone_dim == 768 ? 2 : 1);

    // Trained-register mask: 64 hex chars = 256 coarse-pitch bins, LSB-first
    // nibbles (see rvc_onnx.cpp). Drives auto-octave in the ML thread.
    s.have_register_mask.store(false);
    memset(s.register_mask, 0, sizeof(s.register_mask));
    auto pos = js.find("\"register_mask\":\"");
    if (pos != std::string::npos) {
        pos += 17;
        auto end = js.find('"', pos);
        if (end != std::string::npos && end - pos == 64) {
            for (int i = 0; i < 64; i++) {
                char c = js[pos + (size_t)i];
                int nib = (c >= 'a') ? c - 'a' + 10 : c - '0';
                for (int k = 0; k < 4; k++)
                    s.register_mask[i*4 + k] = (nib >> k) & 1;
            }
            s.have_register_mask.store(true);
        }
    }
}

// ── MIDI export ───────────────────────────────────────────────────────────────

static const char* note_name(int midi) {
    static const char* names[] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    static char buf[16];
    snprintf(buf, sizeof(buf), "%s%d", names[midi % 12], midi / 12 - 1);
    return buf;
}

// SMF format 0, 480 tpq at 120 bpm → 1 second = 960 ticks.
static bool write_midi(const std::string& path, const std::vector<NoteEvent>& notes) {
    std::vector<uint8_t> trk;
    auto vlq = [&](uint32_t v) {
        uint8_t b[4]; int n = 0;
        do { b[n++] = v & 0x7F; v >>= 7; } while (v);
        while (n--) trk.push_back((uint8_t)(b[n] | (n ? 0x80 : 0)));
    };
    // Tempo meta: 500000 us/quarter (120 bpm)
    vlq(0); trk.insert(trk.end(), {0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20});

    // Interleave on/off events sorted by time
    struct Ev { uint32_t tick; uint8_t status, note, vel; };
    std::vector<Ev> evs;
    for (const auto& n : notes) {
        if (n.t_off <= n.t_on || n.note < 0 || n.note > 127) continue;
        evs.push_back({(uint32_t)(n.t_on  * 960.f), 0x90, (uint8_t)n.note, 96});
        evs.push_back({(uint32_t)(n.t_off * 960.f), 0x80, (uint8_t)n.note, 0});
    }
    std::stable_sort(evs.begin(), evs.end(),
                     [](const Ev& a, const Ev& b) { return a.tick < b.tick; });
    uint32_t t = 0;
    for (const auto& e : evs) {
        vlq(e.tick - t); t = e.tick;
        trk.push_back(e.status); trk.push_back(e.note); trk.push_back(e.vel);
    }
    vlq(0); trk.insert(trk.end(), {0xFF, 0x2F, 0x00});  // end of track

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    auto u32 = [&](uint32_t v) {
        uint8_t b[4] = {(uint8_t)(v>>24),(uint8_t)(v>>16),(uint8_t)(v>>8),(uint8_t)v};
        f.write((char*)b, 4);
    };
    auto u16 = [&](uint16_t v) {
        uint8_t b[2] = {(uint8_t)(v>>8),(uint8_t)v};
        f.write((char*)b, 2);
    };
    f.write("MThd", 4); u32(6); u16(0); u16(1); u16(480);
    f.write("MTrk", 4); u32((uint32_t)trk.size());
    f.write((char*)trk.data(), (std::streamsize)trk.size());
    return (bool)f;
}

// ── Use model: export VITS ONNX, configure AppState ─────────────────────────

static void use_model(const std::string& pth_path, AppState& s) {
    ml_thread_stop(&s);
    s.vits_loaded.store(false);
    s.model_state.store(1);           // exporting
    s.loading_path = pth_path;

    uint64_t gen = ++s.export_gen;
    s.export_status.store(1);
    snprintf(s.status_msg, sizeof(s.status_msg), "Loading model\u2026");

    std::thread([pth_path, &s, gen]() {
        PthModel pth = pth_open(pth_path);
        if (!pth.err.empty()) {
            if (gen == s.export_gen.load()) {
                s.export_error = pth.err;
                s.export_status.store(3);
            }
            return;
        }

        int version = 2;
        auto it = pth.tensors.find("enc_p.emb_phone.weight");
        if (it != pth.tensors.end() && !it->second.shape.empty())
            version = (it->second.shape[0] <= 512) ? 1 : 2;
        s.rvc_version.store(version);

        std::string onnx = pth_path.substr(0, pth_path.rfind('.')) + "_vits3.onnx";

        if (!fs::exists(onnx)) {
            std::string err = pth_to_onnx(pth, onnx);
            if (!err.empty()) {
                pth_close(pth);
                if (gen == s.export_gen.load()) {
                    s.export_error = err;
                    s.export_status.store(3);
                }
                return;
            }
        }
        pth_close(pth);
        if (gen != s.export_gen.load()) return;  // superseded by a newer Use click

        s.vits_onnx_path = onnx;
        read_vits_json(onnx, s);
        s.model_path = pth_path;
        s.vits_loaded.store(true);
        s.export_status.store(2);
    }).detach();
}

// ── UI ────────────────────────────────────────────────────────────────────────

static void draw_ui(AppState& s, const ma_device& dev) {
    // Poll export thread
    int exp = s.export_status.load();
    if (exp == 2) {
        if (!s.ml_running.load()) ml_thread_start(&s);
        s.export_status.store(0);
    } else if (exp == 3) {
        s.model_state.store(4);
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
        ImGuiWindowFlags_NoMove    | ImGuiWindowFlags_NoCollapse);

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
        std::string key = dl_key(repo, file);
        HFDownload& dl  = s_dl[key];
        hf_download_poll(dl);

        if (dl.status.load(std::memory_order_acquire) == HFDownload::Status::Done) {
            use_model(dl.out_path, s);
            dl.status.store(HFDownload::Status::Idle, std::memory_order_release);
        }

        std::string card_path = hf_rvc_model_path(repo, file);
        bool installed = hf_rvc_installed(repo, file);
        bool active    = (s.model_path == card_path);
        int  mstate    = s.model_state.load();
        bool loading   = (mstate == 1 || mstate == 2) && (s.loading_path == card_path);

        ImGui::PushID(id);
        ImVec2 cp = ImGui::GetCursorScreenPos();
        float cw = w, ch = 44.f;
        ImDrawList* dl_draw = ImGui::GetWindowDrawList();
        bool hov = ImGui::IsMouseHoveringRect(cp, {cp.x+cw, cp.y+ch});

        dl_draw->AddRectFilled(cp, {cp.x+cw, cp.y+ch},
            hov ? IM_COL32(20,36,32,255) : IM_COL32(14,24,20,255), 4.f);
        ImU32 border = loading   ? IM_COL32(255,200,60,200)
                     : installed ? IM_COL32(30,200,150,180)
                     :             IM_COL32(40,60,50,160);
        dl_draw->AddRect(cp, {cp.x+cw, cp.y+ch}, border, 4.f, 0, 1.f);
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
            const char* state_text;
            ImU32       state_color;
            if (loading) {
                state_text  = "Loading\u2026";
                state_color = IM_COL32(255,200,60,255);
            } else if (active && mstate == 3) {
                state_text  = "Active";
                state_color = IM_COL32(30,220,150,255);
            } else if (active && mstate == 4) {
                state_text  = "Error";
                state_color = IM_COL32(220,80,80,255);
            } else {
                state_text  = "Installed";
                state_color = IM_COL32(60,140,100,180);
            }
            dl_draw->AddText({cp.x+8.f, cp.y+28.f}, state_color, state_text);
            ImGui::SetCursorScreenPos({cp.x+cw-44.f, cp.y+8.f});
            if (ImGui::SmallButton("Use##u")) {
                use_model(card_path, s);
            }
        } else {
            ImGui::SetCursorScreenPos({cp.x+cw-76.f, cp.y+ch/2.f-8.f});
            if (ImGui::SmallButton("Download##d"))
                hf_download_model(repo, file, hf_rvc_model_path(repo, file), dl);
            if (dst == HFDownload::Status::Error) {
                dl_draw->AddText({cp.x+8.f, cp.y+28.f}, IM_COL32(220,80,80,200), "Failed");
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
        if (s_search.results.empty()) ImGui::TextDisabled("No models found.");
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

    // ── Audio device info ─────────────────────────────────────────────────────
    ImGui::TextDisabled("Input:  %s", audio_capture_name(&dev));
    ImGui::TextDisabled("Output: %s", audio_playback_name(&dev));

    ImGui::Separator();

    // ── Controls ─────────────────────────────────────────────────────────────
    {
        float v = s.param_mix.load();
        if (ImGui::SliderFloat("Output Level", &v, 0.0f, 1.0f, "%.2f"))
            s.param_mix.store(v);
    }
    {
        bool easy = s.easy_mode.load();
        if (ImGui::Checkbox("Easy Mode", &easy))
            s.easy_mode.store(easy);
        ImGui::SetItemTooltip("Smooth pitch contour (~250ms damping) — less jitter, more deliberate delivery");
    }
    {
        // Gate: input level meter with the threshold marked, plus slider.
        float rms = s.input_rms.load();
        float thr = s.gate_threshold.load();
        bool  open = s.gate_open.load();
        ImVec2 mp = ImGui::GetCursorScreenPos();
        float mw = w - 110.f, mh = 8.f;
        ImDrawList* mdl = ImGui::GetWindowDrawList();
        auto level_x = [&](float v) {   // log scale, -60dB..0dB
            float db = 20.f * log10f(std::max(v, 1e-4f));
            return mp.x + mw * std::min(1.f, std::max(0.f, (db + 60.f) / 60.f));
        };
        mdl->AddRectFilled(mp, {mp.x + mw, mp.y + mh}, IM_COL32(15, 25, 20, 255), 2.f);
        mdl->AddRectFilled(mp, {level_x(rms), mp.y + mh},
            open ? IM_COL32(30, 220, 150, 255) : IM_COL32(90, 110, 100, 255), 2.f);
        mdl->AddLine({level_x(thr), mp.y - 2.f}, {level_x(thr), mp.y + mh + 2.f},
                     IM_COL32(255, 200, 60, 220), 2.f);
        ImGui::Dummy({mw, mh + 4.f});
        ImGui::SameLine();
        ImGui::TextDisabled(open ? "mic open" : "gated");
        float thr_db = 20.f * log10f(thr);
        ImGui::SetNextItemWidth(w - 110.f);
        if (ImGui::SliderFloat("Gate", &thr_db, -60.f, -20.f, "%.0f dB"))
            s.gate_threshold.store(powf(10.f, thr_db / 20.f));
        ImGui::SetItemTooltip("Input level below this is treated as silence — no synthesis, no output");
    }
    {
        bool ao = s.auto_octave_on.load();
        if (ImGui::Checkbox("Auto Octave", &ao))
            s.auto_octave_on.store(ao);
        ImGui::SetItemTooltip("Shift into the voice model's trained register, read from its emb_pitch weights");
        if (ao && s.have_register_mask.load()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%+d st", s.auto_octave.load());
        } else if (ao && s.ml_running.load()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(no register mask — re-export model)");
        }
    }

    ImGui::Separator();

    // ── Live pitched transcription ────────────────────────────────────────────
    {
        int note = s.live_note.load();
        ImGui::Text("Note: %s", note >= 0 ? note_name(note) : "--");
        ImGui::SameLine();
        float f0 = s.live_f0.load();
        if (f0 > 0.f) ImGui::TextDisabled("%.1f Hz", f0);
        else          ImGui::TextDisabled(" ");

        // Piano roll: last 10 seconds, MIDI 36 (C2) – 96 (C7)
        const float kSpan = 10.f;
        const int   kLo = 36, kHi = 96;
        float now = (float)s.ml_frames.load() * 0.01f;

        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 sz = {w, 90.f};
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p0, {p0.x + sz.x, p0.y + sz.y}, IM_COL32(10, 16, 14, 255), 4.f);

        auto note_y = [&](float n) {
            float u = (n - (float)kLo) / (float)(kHi - kLo);
            return p0.y + sz.y - u * sz.y;
        };
        // Octave gridlines (C rows)
        for (int n = kLo; n <= kHi; n += 12)
            dl->AddLine({p0.x, note_y((float)n)}, {p0.x + sz.x, note_y((float)n)},
                        IM_COL32(30, 46, 40, 120));

        auto draw_bar = [&](float t_on, float t_off, int n, bool active) {
            float x0 = p0.x + (t_on  - (now - kSpan)) / kSpan * sz.x;
            float x1 = p0.x + (t_off - (now - kSpan)) / kSpan * sz.x;
            if (x1 < p0.x || x0 > p0.x + sz.x || n < kLo || n > kHi) return;
            x0 = std::max(x0, p0.x);
            x1 = std::min(x1, p0.x + sz.x);
            float y = note_y((float)n);
            dl->AddRectFilled({x0, y - 2.5f}, {x1, y + 2.5f},
                active ? IM_COL32(80, 255, 190, 255) : IM_COL32(30, 200, 150, 200), 2.f);
        };
        size_t n_notes = 0;
        {
            std::lock_guard<std::mutex> lk(s.notes_mu);
            n_notes = s.notes.size();
            for (auto it = s.notes.rbegin(); it != s.notes.rend(); ++it) {
                if (it->t_off < now - kSpan) break;
                draw_bar(it->t_on, it->t_off, it->note, false);
            }
        }
        if (note >= 0) draw_bar(now - 0.2f, now, note, true);  // live tail

        ImGui::Dummy(sz);
        if (ImGui::SmallButton("Save MIDI") && n_notes > 0) {
            std::vector<NoteEvent> copy;
            {
                std::lock_guard<std::mutex> lk(s.notes_mu);
                copy = s.notes;
            }
            time_t tt = time(nullptr);
            struct tm tmv;
            localtime_r(&tt, &tmv);
            char name[160];
            const char* home = getenv("HOME");
            snprintf(name, sizeof(name), "%s/prism_%04d%02d%02d_%02d%02d%02d.mid",
                     home ? home : ".", tmv.tm_year + 1900, tmv.tm_mon + 1,
                     tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
            snprintf(s.status_msg, sizeof(s.status_msg),
                     write_midi(name, copy) ? "Saved %s" : "MIDI write failed: %s", name);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%zu notes", n_notes);
        ImGui::SameLine();
        ImGui::TextDisabled(s.rmvpe_active.load() ? "RMVPE" : "YIN");
    }

    ImGui::Separator();

    // ── Status bar ────────────────────────────────────────────────────────────
    bool live = s.audio_active.load();
    bool ml   = s.ml_running.load();
    float ms  = s.ml_ms_per_chunk.load();

    ImGui::TextColored(live ? ImVec4{0.2f,1.f,0.2f,1.f} : ImVec4{0.5f,0.5f,0.5f,1.f},
                       live ? "● LIVE" : "○ IDLE");
    ImGui::SameLine();
    if (ml && ms > 0.f) ImGui::Text("%.0fms/chunk", ms);
    else                ImGui::Text("--");
    ImGui::SameLine();

    // Buffer health bar
    if (ml) {
        float fill = (float)s.output_ring.available() / (float)OutputRing::CAP;
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
            fill < 0.1f ? ImVec4{1.f,0.3f,0.3f,1.f} : ImVec4{0.2f,0.8f,0.5f,1.f});
        ImGui::ProgressBar(fill, {60.f, 0.f}, "");
        ImGui::PopStyleColor();
        ImGui::SameLine();
    }

    int mstate = s.model_state.load();
    ImVec4 status_col;
    if (mstate == 1 || mstate == 2) status_col = {1.0f, 0.8f, 0.2f, 1.0f};   // yellow
    else if (mstate == 4)           status_col = {1.0f, 0.3f, 0.3f, 1.0f};   // red
    else                            status_col = {0.85f, 0.85f, 0.85f, 1.0f}; // gray
    ImGui::TextColored(status_col, "%s", s.status_msg);

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

    AppState state;
    ma_device dev;

    if (!audio_init(&state, &dev, "", "")) {
        fprintf(stderr, "Failed to open audio device\n");
        return 1;
    }
    audio_start(&dev, &state);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        draw_ui(state, dev);

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
    audio_uninit(&dev);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
