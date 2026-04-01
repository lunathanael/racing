#pragma once

#include <SDL3/SDL.h>

void spawn_drift_particles(float car_x, float car_y, float car_dir, float speed);
void update_drift_particles();
void render_drift_particles(SDL_Renderer *renderer, float cam_x, float cam_y);