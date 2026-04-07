#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <random>

#ifndef __EMSCRIPTEN__
#include <stdfloat>
#endif

namespace nn
{

#ifdef __EMSCRIPTEN__
using f64_t = double;
using f32_t = float;
using f16_t = __fp16;
#else
using f64_t = std::float64_t;
using f32_t = std::float32_t;
using f16_t = std::float16_t;
#endif

template <class T, size_t size> using Tensor = std::array<T, size>;

namespace rand
{
inline std::mt19937 gen;

template <class... Args> void seed(Args &&...args)
{
    gen.seed(std::forward<Args>(args)...);
}

template <std::integral T> decltype(auto) random_uniform(const T &a, const T &b)
{
    std::uniform_int_distribution<T> dis(a, b);
    return dis(gen);
}

template <std::floating_point T> T random_uniform(T a, T b)
{
    std::uniform_real_distribution<T> dis(a, b);
    return dis(gen);
}
} // namespace rand

template <class T>
concept HasRelu = requires(T t) {
    { t.relu() };
};

template <class T> decltype(auto) relu(T &&v)
{
    if constexpr (HasRelu<T>)
    {
        return std::forward<T>(v).relu();
    }
    else
    {
        return std::max(v, T{});
    }
}

struct ReLUFunc
{
    template <class T> decltype(auto) static inline operator()(T &&v) { return relu(std::forward<T>(v)); }
};

template <class T>
concept HasLeakyRelu = requires(T t) {
    { t.leaky_relu() };
};

template <class T> decltype(auto) leaky_relu(T &&v)
{
    if constexpr (HasLeakyRelu<T>)
    {
        return std::forward<T>(v).leaky_relu();
    }
    else
    {
        constexpr T a = static_cast<T>(1e-2);

        const bool is_pos = v > 0;
        return is_pos ? std::forward<T>(v) : a * std::forward<T>(v);
    }
}

struct LeakyReLUFunc
{
    template <class T> static inline decltype(auto) operator()(T &&v) { return leaky_relu(std::forward<T>(v)); }
};

template <class T, size_t N> auto mae(const Tensor<T, N> &x, const Tensor<T, N> &y)
{
    auto sum = std::transform_reduce(x.cbegin(), x.cend(), y.cbegin(), T{}, std::plus<>{},
                                     [](const auto &a, const auto &b) { return std::abs(a - b); });
    return sum / N;
}

template <class T, size_t N> auto mse(const Tensor<T, N> &x, const Tensor<T, N> &y)
{
    auto sum = std::transform_reduce(x.cbegin(), x.cend(), y.cbegin(), T{}, std::plus<>{},
                                     [](const auto &a, const auto &b) { return (a - b) * (a - b); });
    return sum / N;
}

template <typename T>
concept HasExp = requires(T t) { t.exp(); };

template <class T> decltype(auto) exp(T &&v)
{
    if constexpr (HasExp<T>)
    {
        return std::forward<T>(v).exp();
    }
    else
    {
        return std::exp(std::forward<T>(v));
    }
}

decltype(auto) softmax(auto &&arr)
{
    const auto mx = *std::max_element(arr.cbegin(), arr.cend());
    using T = std::decay_t<decltype(mx)>;
    std::transform(arr.begin(), arr.end(), arr.begin(), [mx](auto &&x) { return std::forward<decltype(x)>(x) - mx; });
    std::transform(arr.begin(), arr.end(), arr.begin(), [](auto &&x) { return exp(std::forward<decltype(x)>(x)); });
    const auto sum = std::accumulate(arr.cbegin(), arr.cend(), T{});
    std::transform(arr.begin(), arr.end(), arr.begin(), [sum](auto &&x) { return std::forward<decltype(x)>(x) / sum; });
    return arr;
}

template <typename T>
concept HasLog = requires(T t) { t.ln(); };

template <class T> decltype(auto) log(T &&v)
{
    if constexpr (HasLog<T>)
    {
        return std::forward<T>(v).ln();
    }
    else
    {
        return std::log(std::forward<T>(v));
    }
}

struct SoftmaxFunc
{
    static inline decltype(auto) operator()(auto &&arr) { return softmax(std::forward<decltype(arr)>(arr)); }
};

decltype(auto) log_softmax(auto &&arr)
{
    const auto mx = *std::max_element(arr.cbegin(), arr.cend());
    using T = std::decay_t<decltype(mx)>;
    const auto sum = std::transform_reduce(arr.cbegin(), arr.cend(), T{ 0 }, std::plus<>{},
                                           [mx](const auto &x) { return exp(std::forward<decltype(x)>(x) - mx); });
    const auto log_sum = log(sum);
    std::transform(arr.begin(), arr.end(), arr.begin(),
                   [mx, log_sum](auto &&x) { return std::forward<decltype(x)>(x) - mx - log_sum; });
    return arr;
}

struct LogSoftmaxFunc
{
    static inline decltype(auto) operator()(auto &&arr) { return log_softmax(std::forward<decltype(arr)>(arr)); }
};

template <class Tp, class Ta, size_t N> Tp cross_entropy_loss(const Tensor<Tp, N> &pred, const Tensor<Ta, N> &actual)
{
    return -std::transform_reduce(actual.cbegin(), actual.cend(), pred.cbegin(), Tp{ 0 }, std::plus<>{},
                                  [](const Ta &a, const Tp &p) { return a * log(p); });
}

template <class Tp, class Ta, size_t N>
Tp cross_entropy_loss_logless(const Tensor<Tp, N> &pred, const Tensor<Ta, N> &actual)
{
    return -std::transform_reduce(actual.cbegin(), actual.cend(), pred.cbegin(), Tp{ 0 }, std::plus<>{},
                                  [](const Ta &a, const Tp &p) { return a * p; });
}
} // namespace nn