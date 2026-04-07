#include "rl/core.h"
#include "rl/utils.h"

#include <catch2/catch_all.hpp>
#include <cmath>

namespace
{
inline nn::f64_t tol = 1e-12;

template <class T, size_t N, size_t K, size_t M>
nn::Tensor<nn::Tensor<T, M>, N> reference_matmul(std::mdspan<T, std::extents<size_t, N, K>> A,
                                                 std::mdspan<T, std::extents<size_t, K, M>> B)
{
    nn::Tensor<nn::Tensor<T, M>, N> out{};
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < M; ++j)
            for (size_t k = 0; k < K; ++k)
                out[i][j] += A[i, k] * B[k, j];
    return out;
}

template <class T, size_t N, size_t K, size_t M>
nn::Tensor<nn::Tensor<T, M>, N> reference_matmul_accum(std::mdspan<T, std::extents<size_t, N, K>> A,
                                                       std::mdspan<T, std::extents<size_t, K, M>> B,
                                                       std::mdspan<T, std::extents<size_t, N, M>> C)
{
    nn::Tensor<nn::Tensor<T, M>, N> out{};
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < M; ++j)
        {
            for (size_t k = 0; k < K; ++k)
                out[i][j] += A[i, k] * B[k, j];
            out[i][j] += C[i, j];
        }
    return out;
}

template <class T, size_t M>
bool approx_equal(const nn::Tensor<T, M> &a, const nn::Tensor<T, M> &b, nn::f64_t epsilon = tol)
{
    for (size_t i = 0; i < M; ++i)
        if constexpr (std::floating_point<T>)
        {
            if (std::abs(a[i] - b[i]) > epsilon)
                return false;
        }
        else if (a[i] != b[i])
            return false;
    return true;
}

template <class T, size_t N, size_t M>
bool approx_equal(const nn::Tensor<nn::Tensor<T, M>, N> &a, const nn::Tensor<nn::Tensor<T, M>, N> &b,
                  nn::f64_t epsilon = tol)
{
    for (size_t i = 0; i < N; ++i)
        if (!approx_equal(a[i], b[i], epsilon))
            return false;
    return true;
}
} // namespace

TEST_CASE("MatMul<int> basic", "[matmul]")
{
    using T = int;
    constexpr size_t N = 2, K = 3, M = 2;
    nn::Tensor<nn::Tensor<T, K>, N> A{ { { 1, 2, 3 }, { 4, 5, 6 } } };
    nn::Tensor<nn::Tensor<T, M>, K> B{ { { 7, 8 }, { 9, 10 }, { 11, 12 } } };

    auto C = nn::matmul(nn::to_mdspan(A), nn::to_mdspan(B));
    auto expected = reference_matmul(nn::to_mdspan(A), nn::to_mdspan(B));

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            REQUIRE(C[i][j] == expected[i][j]);
}

TEST_CASE("MatMul<int> accumulation", "[matmul]")
{
    using T = int;
    constexpr size_t N = 2, K = 2, M = 2;
    nn::Tensor<nn::Tensor<T, K>, N> A{ { { 1, 2 }, { 3, 4 } } };
    nn::Tensor<nn::Tensor<T, M>, K> B{ { { 5, 6 }, { 7, 8 } } };
    nn::Tensor<nn::Tensor<T, M>, N> C{ { { 1, 1 }, { 1, 1 } } };

    auto result = nn::matmul(nn::to_mdspan(A), nn::to_mdspan(B), nn::to_mdspan(C));
    auto expected = reference_matmul_accum(nn::to_mdspan(A), nn::to_mdspan(B), nn::to_mdspan(C));

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            REQUIRE(result[i][j] == expected[i][j]);
}

TEST_CASE("MatMul<float> basic", "[matmul]")
{
    using T = float;
    constexpr size_t N = 2, K = 3, M = 2;
    nn::Tensor<nn::Tensor<T, K>, N> A{ { { 1.f, 2.f, 3.f }, { 4.f, 5.f, 6.f } } };
    nn::Tensor<nn::Tensor<T, M>, K> B{ { { 7.f, 8.f }, { 9.f, 10.f }, { 11.f, 12.f } } };

    auto C = nn::matmul(nn::to_mdspan(A), nn::to_mdspan(B));
    auto expected = reference_matmul(nn::to_mdspan(A), nn::to_mdspan(B));

    REQUIRE(approx_equal(C, expected));
}

TEST_CASE("MatMul<float> accumulation", "[matmul]")
{
    using T = float;
    constexpr size_t N = 2, K = 2, M = 2;
    nn::Tensor<nn::Tensor<T, K>, N> A{ { { 1.f, 2.f }, { 3.f, 4.f } } };
    nn::Tensor<nn::Tensor<T, M>, K> B{ { { 5.f, 6.f }, { 7.f, 8.f } } };
    nn::Tensor<nn::Tensor<T, M>, N> C{ { { 1.f, 1.f }, { 1.f, 1.f } } };

    auto result = nn::matmul(nn::to_mdspan(A), nn::to_mdspan(B), nn::to_mdspan(C));
    auto expected = reference_matmul_accum(nn::to_mdspan(A), nn::to_mdspan(B), nn::to_mdspan(C));

    REQUIRE(approx_equal(result, expected));
}

TEST_CASE("MatMul<double> basic", "[matmul]")
{
    using T = double;
    constexpr size_t N = 2, K = 3, M = 2;
    nn::Tensor<nn::Tensor<T, K>, N> A{ { { 1., 2., 3. }, { 4., 5., 6. } } };
    nn::Tensor<nn::Tensor<T, M>, K> B{ { { 7., 8. }, { 9., 10. }, { 11., 12. } } };

    auto C = nn::matmul(nn::to_mdspan(A), nn::to_mdspan(B));
    auto expected = reference_matmul(nn::to_mdspan(A), nn::to_mdspan(B));

    REQUIRE(approx_equal(C, expected));
}

TEST_CASE("MatMul<double> accumulation", "[matmul]")
{
    using T = double;
    constexpr size_t N = 2, K = 2, M = 2;
    nn::Tensor<nn::Tensor<T, K>, N> A{ { { 1., 2. }, { 3., 4. } } };
    nn::Tensor<nn::Tensor<T, M>, K> B{ { { 5., 6. }, { 7., 8. } } };
    nn::Tensor<nn::Tensor<T, M>, N> C{ { { 1., 1. }, { 1., 1. } } };

    auto result = nn::matmul(nn::to_mdspan(A), nn::to_mdspan(B), nn::to_mdspan(C));
    auto expected = reference_matmul_accum(nn::to_mdspan(A), nn::to_mdspan(B), nn::to_mdspan(C));

    REQUIRE(approx_equal(result, expected));
}

TEST_CASE("MatMul<nn::f64_t> random", "[matmul]")
{
    using T = nn::f64_t;
    constexpr size_t N = 5, K = 4, M = 3;
    nn::Tensor<nn::Tensor<T, K>, N> A{};
    nn::Tensor<nn::Tensor<T, M>, K> B{};

    nn::rand::seed(42);

    for (auto &row : A)
        for (auto &val : row)
            val = nn::rand::random_uniform<T>(-10, 10);

    for (auto &row : B)
        for (auto &val : row)
            val = nn::rand::random_uniform<T>(-10, 10);

    auto expected = reference_matmul(nn::to_mdspan(A), nn::to_mdspan(B));
    auto C = nn::matmul(nn::to_mdspan(A), nn::to_mdspan(B));

    REQUIRE(approx_equal(C, expected, 1e-10));
}