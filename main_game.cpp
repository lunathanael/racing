#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_timer.h>

#include "game/race.h"

#include "render/car.h"
#include "render/common.h"
#include "render/drift.h"
#include "render/hud.h"
#include "render/track.h"

#include <cstdio>
#include <iostream>

// app state
struct AppState
{
    SDL_Window *window;
    SDL_Renderer *renderer;
    Gamestate game;
    Action actions;
    Uint64 last_step;
    bool debug_checkpoints;
    bool map_selector_open;
};

static void scan_key_events(Action &actions)
{
    const bool *state = SDL_GetKeyboardState(nullptr);
    if (state[SDL_SCANCODE_RIGHT] || state[SDL_SCANCODE_D])
        actions.right = true;
    if (state[SDL_SCANCODE_LEFT] || state[SDL_SCANCODE_A])
        actions.left = true;
    if (state[SDL_SCANCODE_UP] || state[SDL_SCANCODE_W])
        actions.accelerate = true;
    if (state[SDL_SCANCODE_DOWN] || state[SDL_SCANCODE_S])
        actions.brake = true;
}

// main loop
SDL_AppResult SDL_AppIterate(void *appstate)
{
    AppState *as = (AppState *)appstate;
    Gamestate &game = as->game;
    Action &actions = as->actions;
    const Uint64 now_ms = SDL_GetTicks();
    static int frame_count = 0;

    actions.clear();

    if (!as->map_selector_open)
    {
        scan_key_events(actions);
        Uint64 now = SDL_GetTicksNS();
        while (now - as->last_step >= STEP_RATE_IN_NANOSECONDS)
        {
            game.step(actions);
            as->last_step += STEP_RATE_IN_NANOSECONDS;
        }
        as->last_step = now;
        frame_count++;
    }

    const auto &car = game.get_car();
    const auto &track = game.get_track();
    const auto &race = game.get_race();

    float cam_x = static_cast<float>(car.get_pos().x);
    float cam_y = static_cast<float>(car.get_pos().y);

    // drift particles
    if (car.is_drifting() && car.get_speed() > 1.0)
    {
        spawn_drift_particles(cam_x, cam_y, static_cast<float>(car.get_dir()), static_cast<float>(car.get_speed()));
    }
    update_drift_particles();

    SDL_SetRenderDrawColor(as->renderer, 25, 25, 30, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(as->renderer);

    if (!as->map_selector_open)
    {
        render_track(as->renderer, track, cam_x, cam_y, as->debug_checkpoints);
        render_drift_particles(as->renderer, cam_x, cam_y);
        render_car(as->renderer, SCREEN_W * 0.5f, SCREEN_H * 0.5f, car.get_dir());

        render_speedometer(as->renderer, static_cast<float>(car.get_speed()));
        render_lap_info(as->renderer, race, now_ms, as->debug_checkpoints);
        render_inputs(as->renderer, actions);
        render_minimap(as->renderer, track, car);
    }
    else
    {
        render_map_selector(as->renderer, game);
    }

    SDL_RenderPresent(as->renderer);

    static Uint64 last_fps_time = SDL_GetTicks();
    if (now_ms - last_fps_time >= 1000)
    {
        std::cout << "FPS: " << frame_count << '\n';
        frame_count = 0;
        last_fps_time = now_ms;
    }

    return SDL_APP_CONTINUE;
}

static const struct
{
    const char *key;
    const char *value;
} extended_metadata[] = { { SDL_PROP_APP_METADATA_URL_STRING, "https://sfin.ae" },
                          { SDL_PROP_APP_METADATA_CREATOR_STRING, "nate" },
                          { SDL_PROP_APP_METADATA_COPYRIGHT_STRING, ":)" },
                          { SDL_PROP_APP_METADATA_TYPE_STRING, "game" } };

SDL_AppResult SDL_AppInit(void **appstate, int argc, char *argv[])
{
    if (!SDL_SetAppMetadata("Sim Racing game", "1.0", "com.sim.Racing"))
        return SDL_APP_FAILURE;
    for (size_t i = 0; i < SDL_arraysize(extended_metadata); i++)
        if (!SDL_SetAppMetadataProperty(extended_metadata[i].key, extended_metadata[i].value))
            return SDL_APP_FAILURE;
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK))
    {
        SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    AppState *as = new AppState{};
    *appstate = as;

    if (!SDL_CreateWindowAndRenderer("Racing", (int)SCREEN_W, (int)SCREEN_H, SDL_WINDOW_RESIZABLE, &as->window,
                                     &as->renderer))
        return SDL_APP_FAILURE;
    SDL_SetRenderLogicalPresentation(as->renderer, (int)SCREEN_W, (int)SCREEN_H, SDL_LOGICAL_PRESENTATION_LETTERBOX);

    // Enable vsync
    SDL_SetRenderVSync(as->renderer, true);

    as->last_step = SDL_GetTicksNS();
    as->game.reset();

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event)
{
    AppState *as = (AppState *)appstate;
    switch (event->type)
    {
    case SDL_EVENT_QUIT:
        return SDL_APP_SUCCESS;
    case SDL_EVENT_KEY_DOWN:
    {
        switch (event->key.scancode)
        {
        case SDL_SCANCODE_Q:
            return SDL_APP_SUCCESS;
        case SDL_SCANCODE_R:
            as->game.reset();
            break;
        case SDL_SCANCODE_M:
            as->map_selector_open ^= 1;
            break;
        case SDL_SCANCODE_C:
            as->debug_checkpoints ^= 1;
            break;
        case SDL_SCANCODE_LEFT:
            if (as->map_selector_open)
            {
                size_t c = as->game.get_maps().size();
                if (c > 0)
                {
                    size_t cur = as->game.get_current_map_idx();
                    as->game.switch_track((cur > 0) ? cur - 1 : c - 1);
                }
            }
            break;
        case SDL_SCANCODE_RIGHT:
            if (as->map_selector_open)
            {
                size_t c = as->game.get_maps().size();
                if (c > 0)
                {
                    size_t cur = as->game.get_current_map_idx();
                    as->game.switch_track((cur + 1) % c);
                }
            }
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void *appstate, SDL_AppResult result)
{
    if (appstate)
    {
        AppState *as = (AppState *)appstate;
        SDL_DestroyRenderer(as->renderer);
        SDL_DestroyWindow(as->window);
        delete as;
    }
}