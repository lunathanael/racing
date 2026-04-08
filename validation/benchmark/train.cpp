#include "rl/core.h"
#include "rl/nn.h"
#include "rl/optim.h"
#include "rl/utils.h"

#include <array>
#include <benchmark/benchmark.h>
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

void BM_TrainMLPSin(benchmark::State &state)
{
    nn::rand::seed(42);

    nn::Sequential<nn::Linear<1, HIDDEN_DIM, T>, nn::ReLU<HIDDEN_DIM, T>, nn::Linear<HIDDEN_DIM, 1, T>> seq{};

    nn::AdamW opt(seq.params(), lr);

    for (auto _ : state)
    {
        for (size_t step = 0; step < TRAIN_STEPS; ++step)
        {
            nn::AutogradNode<T> total_loss{};

            for (size_t i = 0; i < BATCH_SIZE; ++i)
            {
                auto x = nn::rand::random_uniform<T>(-1, 1);
                auto input = nn::Tensor<T, 1>{ x };
                auto pred = seq.forward(input)[0];
                auto actual = target_fn(x);
                auto loss = (pred - actual) * (pred - actual);
                total_loss += loss;
            }
            total_loss /= BATCH_SIZE;

            opt.zero_grad();
            total_loss.backward();
            opt.step();
        }
    }
}

BENCHMARK(BM_TrainMLPSin);