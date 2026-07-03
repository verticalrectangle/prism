#pragma once
// Native PipeWire duplex backend — ported from pop-maker-studio's audio_pw.
//
// The pulse shim quantizes miniaudio's 128-frame request up to ~236/384
// frames per direction (~14 ms round trip for the dry monitor). A native
// pw_stream pair gets the real 128-frame quantum, and both streams tick the
// SAME graph clock, so the capture→playback monitor path is one cycle
// (~2.7 ms) with no drift.
//
// Callbacks run on PipeWire's RT data thread — same rules as a miniaudio
// callback: no locks, no allocations.
#ifdef HAVE_PIPEWIRE
struct AppState;

// Connect the stream pair (mono 48kHz in + out, default devices). Returns
// false if PipeWire is unreachable or the streams fail to reach STREAMING
// within ~2s — caller falls back to miniaudio.
bool audio_pw_start(AppState* s);
void audio_pw_stop();
bool audio_pw_active();
#endif
