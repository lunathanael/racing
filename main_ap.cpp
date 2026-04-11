#include "rl/env.h"
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
    Env env;
    Uint64 last_step;
    bool debug_checkpoints;
    bool map_selector_open;
};

Action get_actions(const Env &env)
{
    constexpr f64_t TURN_SIGHT_DIST = 211.41;
    constexpr f64_t TURN_SPEED = 24.34;
    constexpr f64_t SHARP_TURN_SPEED = 6.25;
    constexpr f64_t MIN_DIST_TO_TURN = 36.94;
    constexpr f64_t MIN_DIST_TO_SHARP_TURN = 57.32;

    Action actions;
    const auto obs = env.get_obs<100>();

    f64_t BRAKE_DIST = 0.5 * obs.vel.x * (obs.vel.x / Car::BREAKING_SPEED);
    f64_t dist{}, disty{};
    for (size_t i = 0; i < obs.future_points.size(); ++i)
    {
        const auto &next = obs.future_points[i];
        dist += next.center.x;
        disty += next.center.y;
        if (dist > BRAKE_DIST)
            break;
        if (std::abs(disty) >= MIN_DIST_TO_TURN)
        {
            if (obs.vel.x > TURN_SPEED)
                actions.brake = true;
            else if (std::abs(disty) > MIN_DIST_TO_SHARP_TURN && obs.vel.x > SHARP_TURN_SPEED)
                actions.brake = true;
            else
                actions.accelerate = true;
            if (dist <= TURN_SIGHT_DIST)
            {
                if (disty > 0)
                    actions.right = true;
                else
                    actions.left = true;
                actions.brake = false;
            }
            return actions;
        }
    }
    actions.accelerate = true;
    return actions;
}

// main loop
SDL_AppResult SDL_AppIterate(void *appstate)
{
    AppState *as = (AppState *)appstate;
    const Gamestate &game = as->env.get_game();
    const Uint64 now_ms = SDL_GetTicks();
    static int frame_count = 0;

    Action actions;
    actions = get_actions(as->env);
    if (!as->map_selector_open)
    {
        Uint64 now = SDL_GetTicksNS();
        if (now - as->last_step >= STEP_RATE_IN_NANOSECONDS)
        {
            as->env.step(actions);
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
    as->env.reset();

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
            as->env.reset();
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
                size_t c = as->env.get_game().get_maps().size();
                if (c > 0)
                {
                    size_t cur = as->env.get_game().get_current_map_idx();
                    as->env.switch_track((cur > 0) ? cur - 1 : c - 1);
                }
            }
            break;
        case SDL_SCANCODE_RIGHT:
            if (as->map_selector_open)
            {
                size_t c = as->env.get_game().get_maps().size();
                if (c > 0)
                {
                    size_t cur = as->env.get_game().get_current_map_idx();
                    as->env.switch_track((cur + 1) % c);
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