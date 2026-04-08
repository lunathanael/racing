#include "rl/ag.h"
#include "rl/core.h"

#include <algorithm>
#include <benchmark/benchmark.h>

template <class T, size_t N, size_t K, size_t M> void BM_MatMul(benchmark::State &state)
{
    nn::rand::seed(42);
    nn::Tensor<nn::Tensor<T, K>, N> A{};
    nn::Tensor<nn::Tensor<T, M>, K> B{};

    for (auto &row : A)
        std::generate(row.begin(), row.end(), []() { return nn::rand::random_uniform<T>(-10, 10); });

    for (auto &row : B)
        std::generate(row.begin(), row.end(), []() { return nn::rand::random_uniform<T>(-10, 10); });

    // warmup
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(A);
        benchmark::DoNotOptimize(B);
    }

    for (auto _ : state)
    {
        auto C = nn::matmul(nn::to_mdspan(A), nn::to_mdspan(B));
        benchmark::DoNotOptimize(C);
    }
}

#define INSTANTIATE_BENCHMARKS(TYPE)                                                                                   \
    BENCHMARK_TEMPLATE(BM_MatMul, TYPE, 1, 5, 5);                                                                      \
    BENCHMARK_TEMPLATE(BM_MatMul, TYPE, 1, 100, 100);                                                                  \
    BENCHMARK_TEMPLATE(BM_MatMul, TYPE, 1, 200, 256);                                                                  \
    BENCHMARK_TEMPLATE(BM_MatMul, TYPE, 25, 25, 25);                                                                   \
    BENCHMARK_TEMPLATE(BM_MatMul, TYPE, 50, 50, 50);                                                                   \
    BENCHMARK_TEMPLATE(BM_MatMul, TYPE, 100, 100, 100);

INSTANTIATE_BENCHMARKS(nn::f32_t)
INSTANTIATE_BENCHMARKS(nn::f64_t)
INSTANTIATE_BENCHMARKS(std::int32_t)
INSTANTIATE_BENCHMARKS(std::int64_t)