#include "rl/core.h"
#include "rl/nn.h"
#include "rl/optim.h"
#include "rl/utils.h"

#include <array>
#include <catch2/catch_all.hpp>
#include <cmath>

namespace
{

using T = nn::f64_t;

inline T target_fn(T x)
{
    return std::sin(x);
}

constexpr size_t HIDDEN_DIM = 16;
constexpr size_t BATCH_SIZE = 32;
constexpr size_t TRAIN_STEPS = 100;
constexpr size_t EVAL_STEPS = 500;
constexpr T LOSS_TOL = 1e-3;
constexpr T lr = 1e-2;

} // namespace

TEST_CASE("Sequential MLP training", "[mlp]")
{
    nn::rand::seed(42);

    nn::Sequential<nn::Linear<1, HIDDEN_DIM, T>, nn::ReLU<HIDDEN_DIM, T>, nn::Linear<HIDDEN_DIM, 1, T>> seq{};

    nn::AdamW opt(seq.params(), lr);

    for (size_t step = 0; step < TRAIN_STEPS; ++step)
    {
        nn::AutogradNode<T> total_loss{};

        for (size_t i = 0; i < BATCH_SIZE; ++i)
        {
            auto x = nn::rand::random_uniform<T>(-1, 1);
            auto input = nn::Tensor<T, 1>{ x };
            auto pred = std::get<0>(seq.forward(input));
            auto actual = target_fn(x);
            auto loss = (pred - actual) * (pred - actual);
            total_loss += loss;
        }
        total_loss /= BATCH_SIZE;

        opt.zero_grad();
        total_loss.backward();
        opt.step();
    }

    T final_loss = 0;
    for (size_t i = 0; i < EVAL_STEPS; ++i)
    {
        auto x = nn::rand::random_uniform<T>(-1, 1);
        auto input = nn::Tensor<T, 1>{ x };
        auto pred = std::get<0>(seq.forward(input));
        auto actual = target_fn(x);
        auto loss = (pred - actual) * (pred - actual);
        final_loss += static_cast<T>(loss);
    }
    final_loss /= EVAL_STEPS;
    REQUIRE(final_loss < LOSS_TOL);
}