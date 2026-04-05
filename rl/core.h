#pragma once

#include "rl/utils.h"

#include <dependencies/mdspan.hpp>

namespace nn
{
template <class Ta, class Tb, size_t N, size_t K, size_t M>
auto matmul(std::mdspan<Ta, std::extents<size_t, N, K>> A, std::mdspan<Tb, std::extents<size_t, K, M>> B)
{
    using out_type = decltype(Ta{} * Tb{});
    Tensor<Tensor<out_type, M>, N> out{};
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < M; ++j)
        {
            for (size_t k = 0; k < K; ++k)
            {
                out[i][j] += A[i, k] * B[k, j];
            }
        }
    }
    return out;
}

template <class Ta, class Tb, size_t N, size_t K, size_t M>
auto matmul(std::mdspan<Ta, std::extents<size_t, N, K>> A, std::mdspan<Tb, std::extents<size_t, K, M>> B,
            std::mdspan<Tb, std::extents<size_t, N, M>> C)
{
    using out_type = decltype(Ta{} * Tb{});
    Tensor<Tensor<out_type, M>, N> out{};
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < M; ++j)
        {
            for (size_t k = 0; k < K; ++k)
            {
                out[i][j] += A[i, k] * B[k, j];
            }
            out[i][j] += C[i, j];
        }
    }
    return out;
}
}