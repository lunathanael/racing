#pragma once

#include "rl/utils.h"

#include <dependencies/mdspan.hpp>

#include <cstddef>
#include <type_traits>

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

template <class Ta, class Tb, class Tc, size_t N, size_t K, size_t M>
auto matmul(std::mdspan<Ta, std::extents<size_t, N, K>> A, std::mdspan<Tb, std::extents<size_t, K, M>> B,
            std::mdspan<Tc, std::extents<size_t, N, M>> C)
{
    using out_type = decltype(Ta{} * Tb{} + Tc{});
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

namespace detail
{

template <class T> struct array_traits
{
    static constexpr bool is_array = false;
};

template <class T, size_t N> struct array_traits<T[N]>
{
    static constexpr bool is_array = true;
    static constexpr size_t size = N;
    using child = T;
};

template <template <class, size_t> class Container, class T, size_t N> struct array_traits<Container<T, N>>
{
    static constexpr bool is_array = true;
    static constexpr size_t size = N;
    using child = T;
};

template <class T, size_t... Dims> consteval auto get_extents()
{
    using PlainT = std::remove_cvref_t<T>;
    if constexpr (array_traits<PlainT>::is_array)
    {
        return get_extents<typename array_traits<PlainT>::child, Dims..., array_traits<PlainT>::size>();
    }
    else
    {
        return std::extents<size_t, Dims...>{};
    }
}

template <class T> constexpr auto get_data(T &t)
{
    if constexpr (requires { t[0]; })
        return get_data(t[0]);
    else
        return &t;
}

template <class T> struct type_wrapper
{
    using type = T;
};

template <size_t Axis, size_t... Exts, size_t... Is>
consteval auto unsqueeze_impl(std::extents<size_t, Exts...>, std::index_sequence<Is...>)
{
    static constexpr std::array<size_t, sizeof...(Exts) + 1> old{ Exts..., 0 };
    return type_wrapper<std::extents<size_t, (Is == Axis ? 1 : old[Is > Axis ? Is - 1 : Is])...>>{};
}

template <size_t Axis, size_t... Exts, size_t... Is>
consteval auto squeeze_impl(std::extents<size_t, Exts...>, std::index_sequence<Is...>)
{
    static constexpr std::array<size_t, sizeof...(Exts) + 1> old{ Exts..., 0 };
    return type_wrapper<std::extents<size_t, old[Is >= Axis ? Is + 1 : Is]...>>{};
}

template <class Extents> struct squeeze_all_traits;

template <> struct squeeze_all_traits<std::extents<size_t>>
{
    using type = std::extents<size_t>;
};

template <size_t First, size_t... Rest> struct squeeze_all_traits<std::extents<size_t, First, Rest...>>
{
    using RestType = typename squeeze_all_traits<std::extents<size_t, Rest...>>::type;

    template <size_t... SqueezedRest> static consteval auto merge(std::extents<size_t, SqueezedRest...>)
    {
        if constexpr (First == 1)
        {
            return type_wrapper<std::extents<size_t, SqueezedRest...>>{};
        }
        else
        {
            return type_wrapper<std::extents<size_t, First, SqueezedRest...>>{};
        }
    }
    using type = typename decltype(merge(RestType{}))::type;
};
} // namespace detail

template <class Array> constexpr auto to_mdspan(Array &arr)
{
    auto extents = detail::get_extents<Array>();
    auto ptr = detail::get_data(arr);

    using BaseType = std::remove_reference_t<decltype(*ptr)>;
    return std::mdspan<BaseType, decltype(extents)>(ptr);
}

template <class Array> constexpr auto to_mdspan(Array &&arr) = delete;

template <size_t Axis, class ElementType, class Extents, class LayoutPolicy, class AccessorPolicy>
constexpr auto unsqueeze(std::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> m)
{
    static_assert(Axis <= Extents::rank(), "unsqueeze Axis out of bounds");
    using NewExtents = typename decltype(detail::unsqueeze_impl<Axis>(
        Extents{}, std::make_index_sequence<Extents::rank() + 1>{}))::type;

    return std::mdspan<ElementType, NewExtents>(m.data_handle());
}

template <size_t Axis, class ElementType, class Extents, class LayoutPolicy, class AccessorPolicy>
constexpr auto squeeze(std::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> m)
{
    static_assert(Axis < Extents::rank(), "squeeze Axis out of bounds");
    static_assert(Extents::static_extent(Axis) == 1, "squeeze target dimension must be of size 1");
    using NewExtents =
        typename decltype(detail::squeeze_impl<Axis>(Extents{}, std::make_index_sequence<Extents::rank() - 1>{}))::type;

    return std::mdspan<ElementType, NewExtents>(m.data_handle());
}

template <class ElementType, class Extents, class LayoutPolicy, class AccessorPolicy>
constexpr auto squeeze(std::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> m)
{
    using NewExtents = typename detail::squeeze_all_traits<Extents>::type;
    return std::mdspan<ElementType, NewExtents>(m.data_handle());
}

} // namespace nn