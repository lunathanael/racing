#include "rl/env.h"
#include "rl/nn.h"
#include "rl/optim.h"

#include "game/race.h"

#include <cstdio>
#include <iostream>

Action get_actions(Env &env, bool force_network = false)
{
    constexpr size_t N_FUTURE_POINTS = 120;
    constexpr size_t POINT_EL = N_FUTURE_POINTS * 2;
    constexpr size_t VEL_EL = 2;

    constexpr size_t OBS_DIM = POINT_EL + VEL_EL;
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
            const auto &obs = ap_env.get_obs<N_FUTURE_POINTS>();
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

    const auto &obs = env.get_obs<N_FUTURE_POINTS>();
    auto input = get_input_arr(obs);

    auto h = seq.forward(input);
    auto nn_probs_steer = pol_steer.forward(h);
    auto nn_probs_drive = pol_drive.forward(h);

    auto act = one_hot_decode(nn_probs_steer, nn_probs_drive);

    size_t score = env.get_game().get_race().get_lap_count() * env.get_game().get_race().get_vertex_count() +
                   env.get_game().get_race().get_next_vertex_idx();
    T reward = (static_cast<T>(score) - static_cast<T>(prev_score)) / 100;

    prev_score = score;

    auto [x, y] = encode_action(act);

    episode.emplace_back(input, x, y, reward);

    return act;
}

int main()
{
    nn::rand::seed(42);

    Env env;
    env.reset();

    while (true)
    {
        Action actions = get_actions(env);
        env.step(actions);
    }

    return 0;
}