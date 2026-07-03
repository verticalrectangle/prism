// test_pipeline — headless reproduction of "hit Use": loads the VITS model the
// same way main.cpp does, starts the real ML thread, feeds 48kHz mono audio
// from stdin (raw f32) into the input ring at real-time rate, and reports
// status / timing / transcription once per second.
//
//   ffmpeg -i clip.flac -f f32le -ac 1 -ar 48000 - | ./prism-test-pipeline <model_vits.onnx> [seconds]
#include "prism.h"
#include "ml_thread.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model_vits.onnx> [seconds] [dump.f32]\n", argv[0]); return 2; }
    int seconds = (argc > 2) ? atoi(argv[2]) : 10;
    FILE* dump = (argc > 3) ? fopen(argv[3], "wb") : nullptr;

    AppState s;
    s.vits_onnx_path = argv[1];

    // Sidecar parse, same fields main.cpp reads
    {
        std::string jp = s.vits_onnx_path.substr(0, s.vits_onnx_path.rfind('.')) + ".json";
        FILE* f = fopen(jp.c_str(), "rb");
        if (f) {
            char buf[512] = {};
            fread(buf, 1, sizeof(buf) - 1, f);
            fclose(f);
            int sr = 40000;
            char* p = strstr(buf, "\"target_sr\":");
            if (p) sr = atoi(p + 12);
            s.output_sr.store(sr);
            p = strstr(buf, "\"register_mask\":\"");
            if (p && strlen(p + 17) >= 64) {
                for (int i = 0; i < 64; i++) {
                    char c = p[17 + i];
                    int nib = (c >= 'a') ? c - 'a' + 10 : c - '0';
                    for (int k = 0; k < 4; k++)
                        s.register_mask[i*4 + k] = (nib >> k) & 1;
                }
                s.have_register_mask.store(true);
            }
            printf("sidecar: sr=%d mask=%d\n", sr, (int)s.have_register_mask.load());
        } else {
            printf("sidecar: MISSING (%s)\n", jp.c_str());
        }
    }
    s.vits_loaded.store(true);

    ml_thread_start(&s);

    // Feed stdin at real-time rate in 128-frame blocks (the audio callback's
    // cadence), draining the output ring like the playback side would.
    std::vector<float> block(128), sink(512);
    auto t_start = std::chrono::steady_clock::now();
    auto next    = t_start;
    long fed = 0, drained = 0;
    int  last_sec = -1;

    while (true) {
        float el = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - t_start).count();
        if (el >= (float)seconds) break;

        size_t got = fread(block.data(), sizeof(float), block.size(), stdin);
        if (got == 0) break;
        s.input_ring.push(block.data(), got);
        fed += (long)got;
        size_t g = s.output_ring.pull(sink.data(), got);
        if (dump && g) fwrite(sink.data(), sizeof(float), g, dump);
        drained += (long)g;

        int sec = (int)el;
        if (sec != last_sec) {
            last_sec = sec;
            size_t n_notes;
            {
                std::lock_guard<std::mutex> lk(s.notes_mu);
                n_notes = s.notes.size();
            }
            printf("[%2ds] status='%s' ml=%.0fms/chunk rmvpe=%d ring=%zu "
                   "oct=%+d f0=%.0f note=%d notes=%zu fed=%ld drained=%ld\n",
                   sec, s.status_msg, s.ml_ms_per_chunk.load(),
                   (int)s.rmvpe_active.load(), s.output_ring.available(),
                   s.auto_octave.load(), s.live_f0.load(), s.live_note.load(),
                   n_notes, fed, drained);
            fflush(stdout);
        }
        next += std::chrono::microseconds(128 * 1000000LL / 48000);
        std::this_thread::sleep_until(next);
    }

    ml_thread_stop(&s);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    printf("final: status='%s' drained=%ld (%.1fs of audio)\n",
           s.status_msg, drained, (float)drained / 48000.f);
    return 0;
}
