#include "game/common.h"
#include "rl/env.h"
#include "rl/nn.h"
#include "rl/optim.h"

#include <cmath>

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
    bool force_nn;
};

Action get_ap_actions(const Env::obs_t &obs)
{
    constexpr f64_t TURN_SIGHT_DIST = 211.41;
    constexpr f64_t TURN_SPEED = 24.34;
    constexpr f64_t SHARP_TURN_SPEED = 6.25;
    constexpr f64_t MIN_DIST_TO_TURN = 36.94;
    constexpr f64_t MIN_DIST_TO_SHARP_TURN = 57.32;

    Action actions;

    f64_t BRAKE_DIST = 0.5 * obs.vel.x * (obs.vel.x / Car::BREAKING_SPEED);
    f64_t dist{}, disty{};
    f64_t px{}, py{};
    for (size_t i = 0; i < obs.future_points.size(); ++i)
    {
        const auto &_next = obs.future_points[i];
        auto next = _next;
        next.x -= px;
        next.y -= py;
        px = next.x;
        py = next.y;
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

Action get_actions(Env &env, bool force_network = false)
{
    constexpr size_t POINT_EL = Env::N_FUTURE_POINTS * 2;
    constexpr size_t VEL_EL = 2;
    constexpr size_t N_ACTIONS = 4;

    constexpr size_t OBS_DIM = POINT_EL + VEL_EL;
    constexpr size_t ACTION_DIM = (1 << N_ACTIONS);
    constexpr size_t HIDDEN_DIM = 16;

    using T = nn::f64_t;
    constexpr T gamma = 0.995;

    constexpr int AP_LAPS = 100;
    constexpr int MAX_TICKS = 2300;

    static nn::Sequential<nn::Linear<OBS_DIM, HIDDEN_DIM, T>, nn::LeakyReLU<HIDDEN_DIM, T>,
                          nn::Linear<HIDDEN_DIM, HIDDEN_DIM, T>, nn::LeakyReLU<HIDDEN_DIM, T>>
        seq("seq.bin");

    static nn::Sequential<nn::Linear<HIDDEN_DIM, 3, T>, nn::Softmax<3, T>> pol_steer("pol_steer.bin");
    static nn::Sequential<nn::Linear<HIDDEN_DIM, 3, T>, nn::Softmax<3, T>> pol_drive("pol_drive.bin");

    static nn::Sequential<nn::Linear<HIDDEN_DIM, 1, T>> val("val.bin");

    static nn::AdamW opt(seq.params(), 1e-4);
    static nn::AdamW opt1a(pol_steer.params(), 1e-4);
    static nn::AdamW opt1b(pol_drive.params(), 1e-4);
    static nn::AdamW opt2(val.params(), 1e-4);

    struct Transition
    {
        std::array<T, OBS_DIM> obs;
        size_t action_idx_steer;
        size_t action_idx_drive;
        T reward;
        Action ap_action;
        bool use_ap;
    };
    static std::vector<Transition> episode;

    static size_t total_laps = 0;
    static size_t prev_score = 1;

    auto encode_action = [](const Action &act)
    {
        int x{}, y{};
        if (act.left)
            x = 1;
        if (act.right)
            x = 2;
        if (act.accelerate)
            y = 1;
        if (act.brake)
            y = 2;
        return std::make_tuple(x, y);
    };

    /*
    normalization
    scheduler
    trajectory buffer
    different algorithm
    optimize code
    gridsearch, random search
    seeding

    try bunch of stuff
    */

    // auto decode_action = [](int action_int) -> Action
    // {
    //     Action act;
    //     act.accelerate = (action_int >> 0) & 1;
    //     act.brake = (action_int >> 1) & 1;
    //     act.left = (action_int >> 2) & 1;
    //     act.right = (action_int >> 3) & 1;
    //     return act;
    // };

    // auto one_hot_encode = [&encode_action](const Action &act)
    // {
    //     int action_int = encode_action(act);
    //     std::array<T, ACTION_DIM> arr{};
    //     arr[action_int] = T(1);
    //     return arr;
    // };

    auto get_input_arr = [](const auto &obs)
    {
        nn::Tensor<T, OBS_DIM> input;
        size_t idx = 0;
        for (const auto &p : obs.future_points)
        {
            input[idx++] = (p.x / 30);
            input[idx++] = (p.y / 30);
        }
        input[idx++] = (obs.vel.x / Car::MAX_SPEED);
        input[idx++] = (obs.vel.y / Car::MAX_SPEED);
        return input;
    };

    auto one_hot_decode = [](const auto &probs_steer, const auto &probs_drive) -> Action
    {
        Action act;
        {
            std::array<T, 3> arr;
            for (int i = 0; i < (int)3; ++i)
            {
                arr[i] = static_cast<T>(probs_steer[i]);
            }
            std::discrete_distribution<int> d(arr.cbegin(), arr.cend());
            auto x = d(nn::rand::gen);

            act.left = (x == 1);
            act.right = (x == 2);
        }
        {
            std::array<T, 3> arr;
            for (int i = 0; i < (int)3; ++i)
            {
                arr[i] = static_cast<T>(probs_drive[i]);
            }
            std::discrete_distribution<int> d(arr.cbegin(), arr.cend());
            auto x = d(nn::rand::gen);

            act.accelerate = (x == 1);
            act.brake = (x == 2);
        }
        return act;
    };

    auto test_lap_time = [&one_hot_decode, &get_input_arr]()
    {
        static Env ap_env;
        ap_env.reset();
        const auto &race = ap_env.get_game().get_race();

        size_t lap_start = race.get_lap_count();
        size_t cnt{};
        while (race.get_lap_count() == lap_start && ++cnt < MAX_TICKS)
        {
            const auto &obs = ap_env.get_obs();
            auto input = get_input_arr(obs);
            auto h = seq.forward(input);
            auto probs_steer = pol_steer.forward(h);
            auto probs_drive = pol_drive.forward(h);

            ap_env.step(one_hot_decode(probs_steer, probs_drive));
        }
        int lm = (int)(static_cast<double>(cnt) / 60.0);
        float ls = static_cast<float>(cnt - lm * 60);
        std::print("Test validation time:{}:{}, {}\n", lm, ls, race.get_next_vertex_idx());
    };

    if ((env.get_game().get_race().get_lap_count() >= 1 || episode.size() >= MAX_TICKS ||
         episode.size() >= env.get_game().get_race().get_next_vertex_idx() * 4 + 200) &&
        !episode.empty())
    {
        int lm = (int)(episode.size() / 60.0);
        float ls = static_cast<float>(episode.size() - lm * 60);
        std::print("Episode time:{}:{}, {}\n", lm, ls, env.get_game().get_race().get_next_vertex_idx());
        env.reset();
        total_laps++;
        prev_score = 1;

        if (total_laps == AP_LAPS)
            std::print("Switching to network!\n");

        const size_t N = episode.size();
        std::vector<T> returns(N);
        T G_t = T(0);
        for (int i = (int)N - 1; i >= 0; --i)
        {
            G_t = episode[i].reward + gamma * G_t;
            returns[i] = G_t;
        }

        T mean = T(0);
        for (auto r : returns)
            mean += r;
        mean /= T(N);

        opt.zero_grad();
        opt1a.zero_grad();
        opt1b.zero_grad();
        opt2.zero_grad();

        auto build_step_loss = [&](size_t i)
        {
            nn::Tensor<T, OBS_DIM> inp;
            for (size_t j = 0; j < OBS_DIM; ++j)
                inp[j] = episode[i].obs[j];

            auto h = seq.forward(inp);
            auto probs_steer = pol_steer.forward(h);
            auto probs_drive = pol_drive.forward(h);
            auto value = val.forward(h)[0];

            T advantage = returns[i] - value.data();

            auto policy_loss = -(probs_steer[episode[i].action_idx_steer] + T(1e-8)).ln() * advantage;
            policy_loss += -(probs_drive[episode[i].action_idx_drive] + T(1e-8)).ln() * advantage;
            auto value_loss = (value - returns[i]) * (value - returns[i]);
            auto step_loss = policy_loss + value_loss * T(0.1);

            return std::make_tuple(step_loss, policy_loss, value_loss);
        };

        auto [total_loss, policy_loss, value_loss] = build_step_loss(0);
        for (size_t i = 1; i < N; ++i)
        {
            auto [t, p, v] = build_step_loss(i);
            total_loss += t;
            policy_loss += p;
            value_loss += v;
        }

        total_loss = total_loss / T(N);
        policy_loss = policy_loss / T(N);
        value_loss = value_loss / T(N);
        total_loss.backward();

        opt.step();
        opt1a.step();
        opt1b.step();
        opt2.step();

        std::print("Episode {}, steps {}, total loss {:.5f}, policy loss {:.5f}, value loss {:.5f}, mean_G {:.3f}",
                   total_laps, N, static_cast<T>(total_loss), static_cast<T>(policy_loss), static_cast<T>(value_loss),
                   mean);
        std::cout << std::endl;

        if (total_laps % 10 == 0)
        {
            seq.params().save("seq.bin");
            pol_steer.params().save("pol_steer.bin");
            pol_drive.params().save("pol_drive.bin");
            val.params().save("val.bin");
            test_lap_time();
        }

        episode.clear();
    }

    const auto &obs = env.get_obs();
    auto input = get_input_arr(obs);

    auto h = seq.forward(input);
    auto nn_probs_steer = pol_steer.forward(h);
    auto nn_probs_drive = pol_drive.forward(h);

    auto ap_action = get_ap_actions(obs);
    bool use_ap = (total_laps < AP_LAPS) && !force_network;

    auto nn_action = one_hot_decode(nn_probs_steer, nn_probs_drive);
    Action act = use_ap ? ap_action : nn_action;

    size_t score = env.get_game().get_race().get_lap_count() * env.get_game().get_race().get_vertex_count() +
                   env.get_game().get_race().get_next_vertex_idx();
    T reward = (static_cast<T>(score) - static_cast<T>(prev_score)) / 100;

    prev_score = score;

    auto [x, y] = encode_action(act);

    episode.push_back({ input, x, y, reward, ap_action, use_ap });

    return act;
}

SDL_AppResult SDL_AppIterate(void *appstate)
{
    AppState *as = (AppState *)appstate;
    const Gamestate &game = as->env.get_game();
    const Uint64 now_ms = SDL_GetTicks();
    static int frame_count = 0;

    Action actions = get_actions(as->env, as->force_nn);

    if (!as->map_selector_open)
    {
        as->env.step(actions);
        as->last_step = SDL_GetTicksNS();
        frame_count++;
    }

    const auto &car = game.get_car();
    const auto &track = game.get_track();
    const auto &race = game.get_race();

    float cam_x = static_cast<float>(car.get_pos().x);
    float cam_y = static_cast<float>(car.get_pos().y);

    if (car.is_drifting() && car.get_speed() > 1.0)
        spawn_drift_particles(cam_x, cam_y, static_cast<float>(car.get_dir()), static_cast<float>(car.get_speed()));
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
    SDL_SetRenderVSync(as->renderer, false);

    as->last_step = SDL_GetTicksNS();
    as->env.reset();

    as->force_nn = true; // default to nn

    nn::rand::seed(42);

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
        switch (event->key.scancode)
        {
        case SDL_SCANCODE_Q:
            return SDL_APP_SUCCESS;
        case SDL_SCANCODE_N:
            as->force_nn ^= 1;
            break;
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