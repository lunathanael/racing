#pragma once

#include <SDL3/SDL.h>

#include "game/track.h"

void render_track(SDL_Renderer *renderer, const Track &track, float cam_x, float cam_y, bool debug_checkpoints);