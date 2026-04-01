#pragma once

#include <algorithm>
#include <iostream>
#include <print>
#include <random>
#include <tuple>

template <class Tuple, class Func> auto transform_tuple(Tuple &&t, Func &&f)
{
    return std::apply([&](auto &&...args)
                      { return std::make_tuple(std::forward<Func>(f)(std::forward<decltype(args)>(args))...); },
                      std::forward<Tuple>(t));
}

template <class... Ts> auto mutate(std::tuple<Ts...> &p, std::mt19937 &rng, double sigma)
{
    std::normal_distribution<> noise(0.0, sigma);
    auto noisify = [&](auto param)
    {
        param += noise(rng);
        return std::clamp<decltype(param)>(param, -1, 1);
    };
    return transform_tuple(p, noisify);
}

template <class... Ts>
auto optimize(auto &&eval, const std::tuple<Ts...> &initial_params, size_t n_iters = 100'000,
              double initial_sigma = 1.0, bool verbose = false)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    auto best = initial_params;
    auto best_score = eval(best);

    if (verbose)
        std::cout << "Initial score: " << best_score << "\n";

    double sigma = initial_sigma;

    for (size_t iter = 0; iter < n_iters; ++iter)
    {
        auto candidate = mutate(best, rng, sigma);

        double score = eval(candidate);

        if (score > best_score)
        {
            best = candidate;
            best_score = score;

            if (verbose)
            {
                std::cout << "\nNEW BEST it: " << iter << '\n';
                std::cout << "score: " << score << "\n";
                std::println("Params: {}", best);
            }
        }

        sigma *= 0.999;
    }
    return best;
}