#pragma once

#include "rl/ag.h"
#include "rl/core.h"
#include "rl/params.h"
#include "rl/utils.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include <dependencies/mdspan.hpp>

namespace nn
{

template <class T, size_t N_PARAMS, bool autograd, class node_t = cond_node_t<T, autograd>> class Layer
{
protected:
    using data_t = _ParamsView<node_t, N_PARAMS>;
    const data_t _raw_data;

    const auto &_data() { return _raw_data; }

public:
    using value_type = T;
    using node_type = node_t;
    static constexpr auto has_autograd = autograd;
    static constexpr auto NUM_EL = N_PARAMS;

    Layer(const data_t &data, bool init = false) : _raw_data{ data } {}
};

template <size_t M, size_t N, class T = f32_t, bool autograd = true, bool use_bias = true,
          class base_t = Layer<T, M * N + (use_bias)*N, autograd>>
class Linear : public base_t
{
    using base_t::_data;
    using typename base_t::data_t;

public:
    static constexpr auto INPUT_DIM = M;
    static constexpr auto OUTPUT_DIM = N;

protected:
    auto _weight()
    {
        auto w = _data().first(M * N).data();
        return std::mdspan(w, std::extents<size_t, M, N>{});
    }
    auto _bias()
        requires(use_bias)
    {
        auto b = _data().last(N).data();
        return std::mdspan(b, std::extents<size_t, 1, N>{});
    }

public:
    Linear(const data_t &data, bool init = true) : base_t{ data }
    {
        static constexpr T k = static_cast<T>(1) / M;
        static constexpr T sqrt_k = std::sqrt(k);

        if (init)
        {
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < M; ++j)
                    _weight()[i, j] = rand::random_uniform(-sqrt_k, sqrt_k);

            if constexpr (use_bias)
            {
                for (size_t i = 0; i < N; ++i)
                    _bias()[0, i] = rand::random_uniform(-sqrt_k, sqrt_k);
            }
        }
    }

    template <class _T, size_t _M> auto forward(const Tensor<_T, _M> &input)
    {
        auto input_span = std::mdspan<const _T, std::extents<size_t, 1, _M>>(input.data());
        auto out = matmul(input_span, _weight(), _bias());
        return out[0];
    }
};

template <class F, size_t M, class T = f32_t, bool autograd = true, class base_t = Layer<T, 0, autograd>>
class ActivationLayer : public base_t
{
    using base_t::_data;

public:
    using base_t::base_t;
    static constexpr auto INPUT_DIM = M;
    static constexpr auto OUTPUT_DIM = M;

    decltype(auto) forward(auto &&input)
    {
        for (size_t i = 0; i < M; ++i)
            input[i] = F::operator()(input[i]);
        return std::forward<decltype(input)>(input);
    }
};

template <size_t M, class T = f32_t, bool autograd = true> using ReLU = ActivationLayer<nn::ReLUFunc, M, T, autograd>;

template <size_t M, class T = f32_t, bool autograd = true>
using LeakyReLU = ActivationLayer<nn::LeakyReLUFunc, M, T, autograd>;

template <class F, size_t M, class T = f32_t, bool autograd = true, class base_t = Layer<T, 0, autograd>>
class FuncLayer : public base_t
{
    using base_t::_data;

public:
    using base_t::base_t;
    static constexpr auto INPUT_DIM = M;
    static constexpr auto OUTPUT_DIM = M;

    decltype(auto) forward(auto &&input) { return F::operator()(std::forward<decltype(input)>(input)); }
};

template <size_t M, class T = f32_t, bool autograd = true> using Softmax = FuncLayer<nn::SoftmaxFunc, M, T, autograd>;

template <size_t M, class T = f32_t, bool autograd = true>
using LogSoftmax = FuncLayer<nn::LogSoftmaxFunc, M, T, autograd>;

template <class... Layers> class Sequential
{
    using first_layer_t = std::tuple_element_t<0, std::tuple<Layers...>>;
    using last_layer_t = std::tuple_element_t<sizeof...(Layers) - 1, std::tuple<Layers...>>;

public:
    static constexpr size_t NUM_EL = (Layers::NUM_EL + ... + 0);

    using value_type = typename first_layer_t::value_type;
    static_assert((std::is_same_v<value_type, typename Layers::value_type> && ...),
                  "All layers in Sequential must be of same type");
    using node_type = typename first_layer_t::node_type;
    static constexpr auto has_autograd = first_layer_t::has_autograd;

    using params_type = Params<value_type, NUM_EL, has_autograd>;

    static constexpr auto INPUT_DIM = first_layer_t::INPUT_DIM;
    static constexpr auto OUTPUT_DIM = last_layer_t::OUTPUT_DIM;

private:
    params_type _params;
    std::tuple<Layers...> _layers;

public:
    Sequential() : _params{}, _layers{ _init_layers(std::index_sequence_for<Layers...>{}) } { _check_all_dims(); }
    Sequential(const std::string &param_file)
        : _params{ param_file }, _layers{ _make_layers(std::index_sequence_for<Layers...>{}) }
    {
        _check_all_dims();
    }

    auto &params() { return _params; }

    decltype(auto) forward(auto &&input) { return _forward_impl(std::forward<decltype(input)>(input)); }

private:
    template <size_t Index = 0> auto _forward_impl(auto &&x)
    {
        if constexpr (Index == sizeof...(Layers))
            return x;
        else
            return _forward_impl<Index + 1>(std::get<Index>(_layers).forward(std::forward<decltype(x)>(x)));
    }

    static consteval void _check_all_dims() { _check_dims<0>(); }

    template <size_t I> static consteval void _check_dims()
    {
        if constexpr (I + 1 < sizeof...(Layers))
        {
            using L1 = std::tuple_element_t<I, std::tuple<Layers...>>;
            using L2 = std::tuple_element_t<I + 1, std::tuple<Layers...>>;
            static_assert(L1::OUTPUT_DIM == L2::INPUT_DIM, "Sequential layer dimension mismatch!");
            _check_dims<I + 1>();
        }
    }

    static consteval auto _offsets()
    {
        std::array<size_t, sizeof...(Layers)> offsets{};
        size_t sum = 0;
        size_t i = 0;
        ((offsets[i++] = sum, sum += Layers::NUM_EL), ...);
        return offsets;
    }

    template <size_t... I> auto _make_layers(std::index_sequence<I...>)
    {
        constexpr auto offsets = _offsets();

        return std::tuple<Layers...>{ Layers(
            _params.template view<offsets[I], std::tuple_element_t<I, std::tuple<Layers...>>::NUM_EL>(), false)... };
    }

    template <size_t... I> auto _init_layers(std::index_sequence<I...>)
    {
        constexpr auto offsets = _offsets();

        return std::tuple<Layers...>{ Layers(
            _params.template view<offsets[I], std::tuple_element_t<I, std::tuple<Layers...>>::NUM_EL>())... };
    }
};
} // namespace nn
