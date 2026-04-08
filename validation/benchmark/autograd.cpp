#include "rl/ag.h"
#include "rl/nn.h"
#include "rl/utils.h"

#include <benchmark/benchmark.h>

template <class T> using node_t = nn::AutogradNode<T>;

template <class T> auto rand_val()
{
    return nn::rand::random_uniform<T>(-10, 10);
}

template <class T, std::size_t depth> node_t<T> make_chain()
{
    node_t<T> x(rand_val<T>());
    for (std::size_t i = 0; i < depth; ++i)
    {
        auto a = rand_val<T>();
        auto b = rand_val<T>();
        x = (x * a + b).relu();
    }
    return x;
}

template <class T, std::size_t depth> node_t<T> make_tree()
{
    if constexpr (depth == 0)
    {
        return node_t<T>(rand_val<T>());
    }
    else
    {
        auto l = make_tree<T, depth - 1>();
        auto r = make_tree<T, depth - 1>();
        return (l * r + l).relu();
    }
}

template <class T, size_t N> node_t<T> make_reduction()
{
    std::array<node_t<T>, N> xs;
    std::array<node_t<T>, N> ws;

    for (size_t i = 0; i < N; ++i)
    {
        xs[i] = rand_val<T>();
        ws[i] = rand_val<T>();
    }

    node_t<T> y = xs[0] * ws[0];
    for (size_t i = 1; i < N; ++i)
    {
        y = y + xs[i] * ws[i];
    }

    return y;
}

template <class T, size_t HIDDEN_DIM> auto make_mlp_like()
{
    nn::Sequential<nn::Linear<1, HIDDEN_DIM, T>, nn::LeakyReLU<HIDDEN_DIM, T>, nn::Linear<HIDDEN_DIM, HIDDEN_DIM, T>,
                   nn::LeakyReLU<HIDDEN_DIM, T>, nn::Linear<HIDDEN_DIM, 1, T>>
        seq;
    return seq;
}

template <class T, size_t N> node_t<T> make_shared()
{
    auto h = make_reduction<T, N>().relu();

    auto a = (h * rand_val<T>()).relu();
    auto b = (h * rand_val<T>()).relu();
    auto c = (h * rand_val<T>()).relu();

    return a + b + c;
}

template <class T, std::size_t depth> void BM_AG_Forward_Chain(benchmark::State &state)
{
    nn::rand::seed(42);
    for (auto _ : state)
    {
        auto y = make_chain<T, depth>();
        benchmark::DoNotOptimize(y);
    }
}

template <class T, std::size_t depth> void BM_AG_Forward_Tree(benchmark::State &state)
{
    nn::rand::seed(42);
    for (auto _ : state)
    {
        auto y = make_tree<T, depth>();
        benchmark::DoNotOptimize(y);
    }
}

template <class T, std::size_t depth> void BM_AG_Backward_Chain(benchmark::State &state)
{
    nn::rand::seed(42);
    auto y = make_chain<T, depth>();
    for (auto _ : state)
    {
        y.zero_grad();
        y.backward();
        benchmark::ClobberMemory();
    }
}

template <class T, std::size_t depth> void BM_AG_Backward_Tree(benchmark::State &state)
{
    nn::rand::seed(42);
    auto y = make_tree<T, depth>();
    for (auto _ : state)
    {
        y.zero_grad();
        y.backward();
        benchmark::ClobberMemory();
    }
}

template <class T, std::size_t depth> void BM_AG_ForwardBackward_Chain(benchmark::State &state)
{
    nn::rand::seed(42);
    for (auto _ : state)
    {
        auto y = make_chain<T, depth>();
        y.backward();
        benchmark::DoNotOptimize(y);
    }
}

template <class T, std::size_t depth> void BM_AG_ForwardBackward_Tree(benchmark::State &state)
{
    nn::rand::seed(42);
    for (auto _ : state)
    {
        auto y = make_tree<T, depth>();
        y.backward();
        benchmark::DoNotOptimize(y);
    }
}

template <class T, std::size_t depth> void BM_AG_ForwardBackward_Reduction(benchmark::State &state)
{
    nn::rand::seed(42);
    for (auto _ : state)
    {
        auto y = make_reduction<T, depth>();
        y.backward();
        benchmark::DoNotOptimize(y);
    }
}

template <class T, std::size_t HIDDEN_DIM> void BM_AG_ForwardBackward_MLP(benchmark::State &state)
{
    nn::rand::seed(42);
    auto seq = make_mlp_like<T, HIDDEN_DIM>();
    for (auto _ : state)
    {
        auto y = seq.forward(nn::Tensor<T, 1>{});
        y[0].backward();
        benchmark::DoNotOptimize(y);
    }
}

template <class T, std::size_t depth> void BM_AG_ForwardBackward_Shared(benchmark::State &state)
{
    nn::rand::seed(42);
    for (auto _ : state)
    {
        auto y = make_shared<T, depth>();
        benchmark::DoNotOptimize(y);
    }
}

#define BENCH_DEPTHS(TYPE)                                                                                             \
    BENCHMARK_TEMPLATE(BM_AG_Forward_Chain, TYPE, 32);                                                                 \
    BENCHMARK_TEMPLATE(BM_AG_Forward_Chain, TYPE, 512);                                                                \
    BENCHMARK_TEMPLATE(BM_AG_Backward_Chain, TYPE, 32);                                                                \
    BENCHMARK_TEMPLATE(BM_AG_Backward_Chain, TYPE, 512);                                                               \
    BENCHMARK_TEMPLATE(BM_AG_Forward_Tree, TYPE, 4);                                                                   \
    BENCHMARK_TEMPLATE(BM_AG_Forward_Tree, TYPE, 8);                                                                   \
    BENCHMARK_TEMPLATE(BM_AG_Backward_Tree, TYPE, 4);                                                                  \
    BENCHMARK_TEMPLATE(BM_AG_Backward_Tree, TYPE, 8);                                                                  \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Chain, TYPE, 8);                                                          \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Chain, TYPE, 64);                                                         \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Chain, TYPE, 512);                                                        \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Tree, TYPE, 2);                                                           \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Tree, TYPE, 4);                                                           \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Tree, TYPE, 8);                                                           \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Reduction, TYPE, 8);                                                      \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Reduction, TYPE, 64);                                                     \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Reduction, TYPE, 512);                                                    \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_MLP, TYPE, 8);                                                            \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_MLP, TYPE, 64);                                                           \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_MLP, TYPE, 512);                                                          \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Shared, TYPE, 8);                                                         \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Shared, TYPE, 64);                                                        \
    BENCHMARK_TEMPLATE(BM_AG_ForwardBackward_Shared, TYPE, 512);

BENCH_DEPTHS(nn::f32_t)
BENCH_DEPTHS(nn::f64_t)