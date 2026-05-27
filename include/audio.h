#pragma once

#include "miniaudio.h"

struct AppState;

bool audio_init(AppState* s, ma_device* dev);
void audio_start(ma_device* dev, AppState* s);
void audio_stop(ma_device* dev, AppState* s);
