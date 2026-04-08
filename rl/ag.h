#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <numbers>
#include <ranges>
#include <unordered_set>
#include <vector>

namespace nn
{
template <class T> class AutogradNode
{
public:
    using value_type = T;

private:
    struct Data
    {
        using prev_arr = std::array<std::shared_ptr<Data>, 2>;
        T _data;
        T _grad;
        std::function<void(const T &)> _backward;
        prev_arr _prev;

        Data(const T &data) : _data{ data }, _grad{}, _backward{}, _prev{} {}
        Data(const T &data, auto &&func, auto &&prev)
            : _data{ data }, _grad{}, _backward{ std::forward<decltype(func)>(func) },
              _prev{ std::forward<decltype(prev)>(prev) }
        {
        }
    };
    std::shared_ptr<Data> _data_ptr;

public:
    AutogradNode(const T &data = {}) : _data_ptr(std::make_shared<Data>(data)) {}

    AutogradNode(const T &data, auto &&func, auto &&prev)
        : _data_ptr{ std::make_shared<Data>(data, std::forward<decltype(func)>(func),
                                            std::forward<decltype(prev)>(prev)) }
    {
    }

    inline const T &data() const { return _data_ptr->_data; }

    inline T &data() { return _data_ptr->_data; }

    inline const T &grad() const { return _data_ptr->_grad; }

    inline T &grad() { return _data_ptr->_grad; }

    auto operator+(const AutogradNode &other) const
    {
        auto self_ptr = _data_ptr;
        auto other_ptr = other._data_ptr;
        auto backward_fn = [self_ptr, other_ptr](const T &local_grad)
        {
            self_ptr->_grad += local_grad;
            other_ptr->_grad += local_grad;
        };
        AutogradNode node(data() + other.data(), std::move(backward_fn),
                          typename Data::prev_arr{ self_ptr, other_ptr });
        return node;
    }

    auto operator+(const T &other) const
    {
        auto self_ptr = _data_ptr;
        auto backward_fn = [self_ptr](const T &local_grad) { self_ptr->_grad += local_grad; };
        AutogradNode node(data() + other, std::move(backward_fn), typename Data::prev_arr{ self_ptr, nullptr });
        return node;
    }

    auto operator-(const AutogradNode &other) const
    {
        auto self_ptr = _data_ptr;
        auto other_ptr = other._data_ptr;
        auto backward_fn = [self_ptr, other_ptr](const T &local_grad)
        {
            self_ptr->_grad += local_grad;
            other_ptr->_grad -= local_grad;
        };
        AutogradNode node(data() - other.data(), std::move(backward_fn),
                          typename Data::prev_arr{ self_ptr, other_ptr });
        return node;
    }

    auto operator-(const T &other) const
    {
        auto self_ptr = _data_ptr;
        auto backward_fn = [self_ptr, other](const T &local_grad) { self_ptr->_grad += local_grad; };
        AutogradNode node(data() - other, std::move(backward_fn), typename Data::prev_arr{ self_ptr, nullptr });
        return node;
    }

    auto operator*(const AutogradNode &other) const
    {
        auto self_ptr = _data_ptr;
        auto other_ptr = other._data_ptr;

        const auto self_data = data();
        const auto other_data = other.data();
        auto backward_fn = [self_ptr, other_ptr, self_data, other_data](const T &local_grad)
        {
            self_ptr->_grad += other_data * local_grad;
            other_ptr->_grad += self_data * local_grad;
        };
        AutogradNode node(data() * other.data(), std::move(backward_fn),
                          typename Data::prev_arr{ self_ptr, other_ptr });
        return node;
    }

    auto operator*(const T &other) const
    {
        auto self_ptr = _data_ptr;

        auto backward_fn = [self_ptr, other](const T &local_grad) { self_ptr->_grad += other * local_grad; };
        AutogradNode node(data() * other, std::move(backward_fn), typename Data::prev_arr{ self_ptr, nullptr });
        return node;
    }

    auto operator/(const AutogradNode &other) const
    {
        auto self_ptr = _data_ptr;
        auto other_ptr = other._data_ptr;

        const auto self_data = data();
        const auto other_data = other.data();
        auto backward_fn = [self_ptr, other_ptr, self_data, other_data](const T &local_grad)
        {
            const auto inv = T{ 1 } / other_data;
            self_ptr->_grad += inv * local_grad;
            other_ptr->_grad += -inv * inv * self_data * local_grad;
        };
        AutogradNode node(data() / other.data(), std::move(backward_fn),
                          typename Data::prev_arr{ self_ptr, other_ptr });
        return node;
    }

    auto operator/(const T &other) const
    {
        auto self_ptr = _data_ptr;

        auto backward_fn = [self_ptr, other](const T &local_grad) { self_ptr->_grad += local_grad / other; };
        AutogradNode node(data() / other, std::move(backward_fn), typename Data::prev_arr{ self_ptr, nullptr });
        return node;
    }

    auto ln() const
    {
        auto self_ptr = _data_ptr;

        const auto self_data = data();

        auto backward_fn = [self_ptr, self_data](const T &local_grad) { self_ptr->_grad += local_grad / self_data; };
        AutogradNode node(std::log(data()), std::move(backward_fn), typename Data::prev_arr{ self_ptr, nullptr });
        return node;
    }

    static auto pow(const T &base, const AutogradNode &exp_node)
    {
        auto exp_ptr = exp_node._data_ptr;

        const auto pow_val = std::pow(base, exp_node.data());
        auto backward_fn = [exp_ptr, base, pow_val](const T &local_grad)
        { exp_ptr->_grad += local_grad * pow_val * std::log(base); };
        AutogradNode node(pow_val, std::move(backward_fn), typename Data::prev_arr{ exp_ptr, nullptr });
        return node;
    }

    auto exp() const
    {
        constexpr T e_v = std::numbers::e_v<T>;
        return pow(e_v, *this);
    }

    auto operator^(const AutogradNode &exp_node) const
    {
        auto self_ptr = _data_ptr;
        auto exp_ptr = exp_node._data_ptr;

        const auto self_data = data();
        const auto exp_data = exp_node.data();

        const auto pow_val = std::pow(data(), exp_data - T{ 1 });

        auto backward_fn = [self_ptr, exp_ptr, self_data, exp_data, pow_val](const T &local_grad)
        {
            self_ptr->_grad += local_grad * exp_data * pow_val;
            exp_ptr->_grad += local_grad * pow_val * self_data * std::log(self_data);
        };
        AutogradNode node(pow_val * data(), std::move(backward_fn), typename Data::prev_arr{ self_ptr, exp_ptr });
        return node;
    }

    auto operator^(const T &other) const
    {
        auto self_ptr = _data_ptr;
        const auto pow_val = std::pow(data(), other - T{ 1 });

        auto backward_fn = [self_ptr, other, pow_val](const T &local_grad)
        { self_ptr->_grad += local_grad * other * pow_val; };
        AutogradNode node(pow_val * data(), std::move(backward_fn), typename Data::prev_arr{ self_ptr, nullptr });
        return node;
    }

    auto operator+=(const auto &other) { return *this = *this + other; }

    auto operator-=(const auto &other) { return *this = *this - other; }

    auto operator*=(const auto &other) { return *this = *this * other; }

    auto operator/=(const auto &other) { return *this = *this / other; }

    auto operator^=(const auto &other) { return *this = *this ^ other; }

    auto operator-() const { return *this * -1; }

    auto relu() const
    {
        auto self_ptr = _data_ptr;

        const auto self_data = data();
        auto backward_fn = [self_ptr, self_data](const T &local_grad)
        { self_ptr->_grad += local_grad * (self_data > 0); };
        AutogradNode node(std::max(T{ 0 }, data()), std::move(backward_fn),
                          typename Data::prev_arr{ self_ptr, nullptr });
        return node;
    }

    auto leaky_relu() const
    {
        constexpr T a = static_cast<T>(1e-2);
        auto self_ptr = _data_ptr;

        const bool is_pos = data() > 0;
        const T val = is_pos ? data() : a * data();

        auto backward_fn = [self_ptr, is_pos](const T &local_grad)
        { self_ptr->_grad += local_grad * (is_pos ? T{ 1 } : a); };

        return AutogradNode(val, std::move(backward_fn), typename Data::prev_arr{ self_ptr, nullptr });
    }

    void zero_grad()
    {
        std::unordered_set<decltype(_data_ptr)> vis;
        auto dfs = [&](this auto &&self, auto curr) -> void
        {
            vis.insert(curr);
            curr->_grad = 0;
            for (auto &&child : curr->_prev)
            {
                if (child && !vis.contains(child))
                    self(child);
            }
        };
        dfs(_data_ptr);
    }

    void backward()
    {
        std::unordered_set<decltype(_data_ptr)> vis;
        std::vector<decltype(_data_ptr)> v;
        auto dfs = [&](this auto &&self, auto curr) -> void
        {
            vis.insert(curr);
            for (auto &&child : curr->_prev)
            {
                if (child && !vis.contains(child))
                    self(child);
            }
            v.emplace_back(std::move(curr));
        };
        dfs(_data_ptr);

        grad() = 1;
        for (auto &&node : v | std::views::reverse)
        {
            if (node->_backward)
                node->_backward(node->_grad);
        }
    }

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

} // namespace nn