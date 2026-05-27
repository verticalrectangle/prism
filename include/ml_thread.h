#pragma once

struct AppState;
struct PrismModel;

void ml_thread_start(AppState* s, PrismModel* model);
void ml_thread_stop(AppState* s);
