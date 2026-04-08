#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <variant>
#include <vector>

namespace nn
{

template <class T> class AutogradTape;

template <class T> inline thread_local AutogradTape<T> global_tape;

template <class T> class AutogradNode
{
    inline thread_local static auto &_tape = global_tape<T>;

public:
    using value_type = T;

private:
    size_t _node_idx;

    decltype(auto) _node() { return _tape[_node_idx]; }

    decltype(auto) _node() const { return _tape[_node_idx]; }

    auto _build_node(const T &new_data, const T &local) const
    {
        return AutogradNode(new_data, std::array<T, 2>{ local, T{} },
                            std::array<size_t, 2>{ _node_idx, AutogradTape<T>::NULL_IDX });
    }

    auto _build_node(const T &new_data, const T &local, const T &other_local, const AutogradNode &other) const
    {
        return AutogradNode(new_data, std::array<T, 2>{ local, other_local },
                            std::array<size_t, 2>{ _node_idx, other._node_idx });
    }

public:
    AutogradNode(const T &data = {}) : _node_idx{ _tape.emplace(data) } {}

    AutogradNode(const T &data, auto &&local, auto &&prev_idx)
        : _node_idx{ _tape.emplace(data, std::forward<decltype(local)>(local),
                                   std::forward<decltype(prev_idx)>(prev_idx)) }
    {
    }

    AutogradNode(const AutogradNode &) = delete;
    AutogradNode &operator=(const AutogradNode &) = delete;

    AutogradNode(AutogradNode &&) = default;
    AutogradNode &operator=(AutogradNode &&) = default;

    inline const T &data() const { return _node().data; }

    inline T &data() { return _node().data; }

    inline const T &grad() const { return _node().grad; }

    inline T &grad() { return _node().grad; }

    auto operator+(const AutogradNode &other) const
    {
        return _build_node(data() + other.data(), T{ 1 }, T{ 1 }, other);
    }

    auto operator+(const T &other) const { return _build_node(data() + other, T{ 1 }); }

    auto operator-(const AutogradNode &other) const
    {
        return _build_node(data() - other.data(), T{ 1 }, T{ -1 }, other);
    }

    auto operator-(const T &other) const { return _build_node(data() - other, T{ 1 }); }

    auto operator*(const AutogradNode &other) const
    {
        return _build_node(data() * other.data(), T{ other.data() }, T{ data() }, other);
    }

    auto operator*(const T &other) const { return _build_node(data() * other, T{ other }); }

    auto operator/(const AutogradNode &other) const
    {
        const auto inv = T{ 1 } / other.data();
        return _build_node(data() / other.data(), inv, -inv * inv * data(), other);
    }

    auto operator/(const T &other) const { return _build_node(data() / other, T{ 1 } / other); }

    auto ln() const { return _build_node(std::log(data()), T{ 1 } / data()); }

    static auto pow(const T &base, const AutogradNode &exp_node)
    {
        const auto pow_val = std::pow(base, exp_node.data());
        return exp_node._build_node(pow_val, pow_val * std::log(base));
    }

    auto exp() const
    {
        constexpr T e_v = std::numbers::e_v<T>;
        return pow(e_v, *this);
    }

    auto operator^(const AutogradNode &exp_node) const
    {
        const auto pow_val = std::pow(data(), exp_node.data() - T{ 1 });
        return _build_node(pow_val * data(), exp_node.data() * pow_val, pow_val * data() * std::log(data()), exp_node);
    }

    auto operator^(const T &other) const
    {
        const auto pow_val = std::pow(data(), other - T{ 1 });
        return _build_node(pow_val * data(), other * pow_val);
    }

    auto &operator+=(const auto &other) { return *this = *this + other; }

    auto &operator-=(const auto &other) { return *this = *this - other; }

    auto &operator*=(const auto &other) { return *this = *this * other; }

    auto &operator/=(const auto &other) { return *this = *this / other; }

    auto &operator^=(const auto &other) { return *this = *this ^ other; }

    auto operator-() const { return *this * -1; }

    auto relu() const { return _build_node(std::max(T{ 0 }, data()), T{ static_cast<T>(data() > 0) }); }

    auto leaky_relu() const
    {
        constexpr T a = static_cast<T>(1e-2);

        const bool is_pos = data() > 0;
        const T val = is_pos ? data() : a * data();

        return _build_node(val, is_pos ? T{ 1 } : a);
    }

    void zero_grad() { grad() = 0; }

    void backward() { _tape.backward(_node_idx); }

    explicit operator T() const { return data(); }

    std::strong_ordering operator<=>(const AutogradNode &other) const
    {
        if (data() < other.data())
            return std::strong_ordering::less;
        if (data() > other.data())
            return std::strong_ordering::greater;
        return std::strong_ordering::equal;
    }

    auto detach() const { return AutogradNode(data()); }
};

template <class T> inline decltype(auto) operator+(const T &lhs, const AutogradNode<T> &rhs)
{
    return rhs + lhs;
}

template <class T> inline decltype(auto) operator-(const T &lhs, const AutogradNode<T> &rhs)
{
    return -rhs + lhs;
}

template <class T> inline decltype(auto) operator*(const T &lhs, const AutogradNode<T> &rhs)
{
    return rhs * lhs;
}

template <class T> inline decltype(auto) operator/(const T &lhs, const AutogradNode<T> &rhs)
{
    return (rhs ^ (-1)) * lhs;
}

template <class T> inline decltype(auto) operator^(const T &lhs, const AutogradNode<T> &rhs)
{
    return rhs.pow(lhs);
}

template <class T, bool autograd> using cond_node_t = std::conditional_t<autograd, AutogradNode<T>, T>;

template <class T> class AutogradTape
{
    inline static constexpr bool isDebug =
#ifdef NDEBUG
        false;
#else
        true;
#endif
    struct TapeNode
    {
        T data;
        T grad;
        std::array<T, 2> local;
        std::array<size_t, 2> prev_idx;

        TapeNode(const T &data_ = {}) : data{ data_ }, grad{}, local{}, prev_idx{} {}
        TapeNode(const T &data_, auto &&local_, auto &&prev_idx_)
            : data{ data_ }, grad{}, local{ std::forward<decltype(local_)>(local_) },
              prev_idx{ std::forward<decltype(prev_idx_)>(prev_idx_) }
        {
        }
    };

    std::vector<TapeNode> _st;

    size_t _static_size;

public:
    inline static constexpr size_t NULL_IDX = 0;

    AutogradTape() : _st(1), _static_size{ 1 } {}

    template <class... Args> size_t emplace(Args &&...args)
    {
        _st.emplace_back(std::forward<Args>(args)...);
        return size() - 1;
    }

    decltype(auto) size() const { return _st.size(); }

    decltype(auto) operator[](size_t idx)
    {
        assert(idx < size());
        return _st[idx];
    }

    void zero_grad()
    {
        for (auto &node : _st)
        {
            node.grad = 0;
        }
    }

    void backward(size_t node_idx = NULL_IDX)
    {
        if (_st.empty())
            return;

        (*this)[node_idx].grad = 1;
        for (; node_idx > _static_size; --node_idx)
        {
            const auto &node = (*this)[node_idx];
            for (size_t i = 0; i < 2; ++i)
            {
                const auto &prev_idx = node.prev_idx[i];
                if (prev_idx == NULL_IDX)
                    break;
                (*this)[prev_idx].grad += node.grad * node.local[i];
            }
        }
        clear();
    }

    void lock() { _static_size = std::max(_static_size, size()); }

    void clear()
    {
        if constexpr (isDebug)
        {
            std::cerr << "INFO: Calling clear on Autograd Tape on " << (size() - _static_size) << " artifacts"
                      << std::endl;
        }
        _st.resize(_static_size);
    }

    void reset()
    {
        _st.resize(1);
        _static_size = size();
    }
};

} // namespace nn