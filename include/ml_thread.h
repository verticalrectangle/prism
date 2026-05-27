#pragma once

struct AppState;
struct HubertModel;
struct RvcModel;

void ml_thread_start(AppState* s, HubertModel* hubert, RvcModel* rvc);
void ml_thread_stop(AppState* s);
