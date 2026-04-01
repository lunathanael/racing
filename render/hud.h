#pragma once

#include "game/car.h"
#include "game/race.h"
#include "game/track.h"

#include <SDL3/SDL.h>

void draw_hud_string(SDL_Renderer *renderer, const char *str, float x, float y, float cw, float ch, float sp);

void render_speedometer(SDL_Renderer *renderer, float speed);
void render_lap_info(SDL_Renderer *renderer, const RaceState &race, uint64_t now_ms, bool debug_checkpoints);
void render_minimap(SDL_Renderer *renderer, const Track &track, const Car &car);
void render_inputs(SDL_Renderer *renderer, const Action &actions);
void render_map_selector(SDL_Renderer *renderer, const Gamestate &game);