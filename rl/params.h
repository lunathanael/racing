#pragma once

#include "rl/ag.h"
#include "rl/utils.h"

#include <array>
#include <fstream>
#include <print>
#include <span>
#include <string>

namespace nn
{

template <class T, size_t N_PARAMS> using _Params = std::array<T, N_PARAMS>;
template <class T, size_t N_PARAMS> using _ParamsView = std::span<T, N_PARAMS>;

template <class T, size_t N_PARAMS, bool autograd, class node_t = cond_node_t<T, autograd>,
          class base_t = _Params<node_t, N_PARAMS>>
class Params : protected base_t
{
public:
    template <class... Args> Params(Args &&...args) : base_t(std::forward<Args>(args)...) { global_tape<T>.lock(); }

    Params(const std::string &file)
    {
        bool loaded = load(file);
        if (!loaded)
        {
            std::print("Could not load parameters from {}, parameters uninitalized.\n", file);
        }
    }

    static constexpr auto NUM_EL = N_PARAMS;

    bool save(const std::string &file) const
    {

        std::ofstream outfile(file, std::ios::binary | std::ios::out);
        if (!outfile.is_open())
            return false;

        if constexpr (autograd)
        {
            std::array<T, N_PARAMS> buf;
            std::transform(base_t::cbegin(), base_t::cend(), buf.begin(),
                           [](const node_t &node) { return node.data(); });
            outfile.write(reinterpret_cast<const char *>(buf.data()), N_PARAMS * sizeof(T));
        }
        else
        {
            outfile.write(reinterpret_cast<const char *>(base_t::data()), N_PARAMS * sizeof(T));
        }
        return true;
    }

    bool load(const std::string &file)
    {
        std::ifstream infile(file, std::ios::binary | std::ios::in);
        if (!infile.is_open())
            return false;
        if constexpr (autograd)
        {
            std::array<T, N_PARAMS> buf;
            infile.read(reinterpret_cast<char *>(buf.data()), N_PARAMS * sizeof(T));
            for (size_t i = 0; i < N_PARAMS; ++i)
                (*this)[i].data() = buf[i];
        }
        else
        {
            infile.read(reinterpret_cast<char *>(base_t::data()), N_PARAMS * sizeof(T));
        }
        return true;
    }

    auto params() { return std::span<node_t, N_PARAMS>(base_t::begin(), base_t::end()); }

    template <size_t offset, size_t count> auto view()
    {
        static_assert(offset + count <= N_PARAMS, "view out of range");
        return params().template subspan<offset, count>();
    }

    void zero_grad()
    {
        if constexpr (autograd)
        {
            for (auto &p : base_t::data())
                p.grad() = 0;
        }
    }
};
} // namespace nn