#pragma once

#include "rl/ag.h"
#include "rl/params.h"
#include "rl/utils.h"

#include <array>
#include <span>
#include <variant>

namespace nn
{

template <class T, size_t N_PARAMS> class Optim
{
public:
    using value_type = T;
    using node_type = AutogradNode<T>;
    using params_type = Params<T, N_PARAMS, true>;

    Optim(params_type &params) : _params{ params } {}

    void zero_grad()
    {
        for (auto &p : _params.template view<0, N_PARAMS>())
            p.grad() = 0;
    }

protected:
    params_type &_params;

    decltype(auto) params() { return _params.params(); }
};

template <class T, size_t N_PARAMS, T momentum = static_cast<T>(0.9), T dampening = {}, T weight_decay = {},
          bool nesterov = false, bool maximize = false, class base_t = Optim<T, N_PARAMS>>
class SGD : public base_t
{
    using base_t::_params;

    static constexpr bool use_momentum = (momentum != 0);

    T _lr;
    using buffer_t = std::conditional_t<use_momentum, std::array<T, N_PARAMS>, std::monostate>;
    buffer_t _buffer;

public:
    SGD(base_t::params_type &params, const auto &lr = static_cast<T>(1e-3))
        : base_t{ params }, _lr{ static_cast<T>(lr) }, _buffer{}
    {
    }

    void step()
    {
        auto &&params = _params.params();
        for (size_t i = 0; i < N_PARAMS; ++i)
        {
            auto &p = params[i];
            T g_t = (maximize) ? -p.grad() : p.grad();
            if constexpr (weight_decay != 0)
            {
                g_t += weight_decay * p.data();
            }
            if constexpr (use_momentum)
            {
                auto &b_t = _buffer[i];
                b_t = static_cast<T>(momentum * b_t + (1 - dampening) * g_t);
                if constexpr (nesterov)
                    g_t += momentum * b_t;
                else
                    g_t = b_t;
            }
            p.data() -= _lr * g_t;
        }
    }

    void set_lr(const auto &lr) { _lr = lr; }
};

template <class T, size_t N_PARAMS> SGD(Params<T, N_PARAMS, true> &) -> SGD<T, N_PARAMS>;

template <class T, size_t N_PARAMS> SGD(Params<T, N_PARAMS, true> &, const auto &lr) -> SGD<T, N_PARAMS>;

template <class T, size_t N_PARAMS, T weight_decay = {}, bool decoupled = false, bool amsgrad = false,
          bool maximize = false, T beta_0 = static_cast<T>(0.9), T beta_1 = static_cast<T>(0.999),
          T eps = static_cast<T>(1e-8), class base_t = Optim<T, N_PARAMS>>
class Adam : public base_t
{
    using base_t::_params;

    T _lr;
    std::array<T, N_PARAMS> _first, _second;
    std::conditional_t<amsgrad, std::array<T, N_PARAMS>, std::monostate> _v_max;
    T b_0, b_1;

public:
    Adam(base_t::params_type &params, const auto &lr = static_cast<T>(1e-3))
        : base_t{ params }, _lr{ static_cast<T>(lr) }, _first{}, _second{}, _v_max{}, b_0{ 1 }, b_1{ 1 }
    {
    }

    void step()
    {
        b_0 *= beta_0;
        b_1 *= beta_1;

        auto params = _params.params();
        for (size_t i = 0; i < N_PARAMS; ++i)
        {
            auto &p = params[i];
            T g_t = (maximize) ? -p.grad() : p.grad();

            if constexpr (!decoupled && weight_decay != 0)
            {
                g_t += weight_decay * p.data();
            }

            auto &m_t = _first[i];
            auto &v_t = _second[i];

            m_t = (beta_0 * m_t) + (1 - beta_0) * g_t;
            v_t = (beta_1 * v_t) + (1 - beta_1) * g_t * g_t;

            T bm_t = m_t / (1 - b_0);
            T bv_t;

            if constexpr (amsgrad)
            {
                auto &_v_max_t = _v_max[i];
                _v_max_t = std::max(_v_max_t, v_t);
                bv_t = _v_max_t / (1 - b_1);
            }
            else
            {
                bv_t = v_t / (1 - b_1);
            }

            p.data() -= _lr * bm_t / (std::sqrt(bv_t) + eps);

            if constexpr (decoupled)
            {
                p.data() *= (1 - (_lr * weight_decay));
            }
        }
    }

    void set_lr(const auto &lr) { _lr = lr; }
};

template <class T, size_t N_PARAMS> Adam(Params<T, N_PARAMS, true> &) -> Adam<T, N_PARAMS>;

template <class T, size_t N_PARAMS> Adam(Params<T, N_PARAMS, true> &, const auto &lr) -> Adam<T, N_PARAMS>;

template <class T, size_t N_PARAMS, T weight_decay = static_cast<T>(0.01), bool amsgrad = false, bool maximize = false,
          T beta_0 = static_cast<T>(0.9), T beta_1 = static_cast<T>(0.999), T eps = static_cast<T>(1e-8),
          class base_t = Adam<T, N_PARAMS, weight_decay, true, amsgrad, maximize, beta_0, beta_1, eps>>
class AdamW : public base_t
{
public:
    using base_t::base_t;
    AdamW(base_t::params_type &params) : base_t(params, T{ 1e-3 }) {}
};

template <class T, size_t N_PARAMS> AdamW(Params<T, N_PARAMS, true> &) -> AdamW<T, N_PARAMS>;

template <class T, size_t N_PARAMS> AdamW(Params<T, N_PARAMS, true> &, const auto &lr) -> AdamW<T, N_PARAMS>;

} // namespace nn
