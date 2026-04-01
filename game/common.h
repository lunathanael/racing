#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>

#ifdef __EMSCRIPTEN__
using f64_t = double;
using f32_t = float;
#else
#include <stdfloat>
using f64_t = std::float64_t;
using f32_t = std::float32_t;
#endif

template <class T> struct vec
{
    T x, y;

    vec operator-(const vec &other) const { return { x - other.x, y - other.y }; }

    vec operator+(const vec &other) const { return { x + other.x, y + other.y }; }

    template <typename float_t> vec rotate(const float_t &rad) const
    {
        const auto c = std::cos(rad);
        const auto s = std::sin(rad);
        return { (x * c) + (y * s), -(x * s) + (y * c) };
    }
};

inline constexpr f64_t PI_V = std::numbers::pi_v<f64_t>;

inline constexpr size_t TICKS_PER_SECOND = 60;
inline constexpr f64_t SECONDS_PER_TICK = 1.0f / TICKS_PER_SECOND;
inline constexpr size_t STEP_RATE_IN_NANOSECONDS = static_cast<size_t>(1e9 / TICKS_PER_SECOND);