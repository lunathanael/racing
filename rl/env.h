#pragma once

#include "game/common.h"
#include "game/race.h"

#include <algorithm>
#include <array>

class Env
{
public:
    using float_t = f64_t;

    struct point_t
    {
        vec<float_t> center; //, left_edge, right_edge;
    };
    template <size_t N_FUTURE_POINTS> struct obs_t
    {
        using points_t = std::array<point_t, N_FUTURE_POINTS>;
        vec<float_t> vel;
        points_t future_points;
    };

private:
    Gamestate game;

    auto get_relative_vel() const
    {
        const auto &car = game.get_car();
        const auto &vel = car.get_vel();
        const auto &dir = car.get_dir();

        return vel.rotate(dir);
    }

    template <size_t N_FUTURE_POINTS> auto get_relative_points() const
    {
        typename obs_t<N_FUTURE_POINTS>::points_t points;
        const auto &race = game.get_race();
        const auto &track = game.get_track();
        const auto &car = game.get_car();

        const auto &next_point = race.get_next_vertex_idx();
        const auto &vertices = track.get_vertices();
        const auto &n_points = vertices.size();
        const auto &pos = car.get_pos();
        const auto &dir = car.get_dir();

        std::for_each(points.begin(), points.end(),
                      [&](auto &p)
                      {
                          const auto idx = &p - points.data();
                          const auto current_idx = (next_point + idx) % n_points;
                          const auto prev_idx = (next_point + idx - 1) % n_points;
                          const auto &prev = (idx == 0) ? pos : vertices[prev_idx].center;
                          const auto &current = vertices[current_idx];
                          p.center = (current.center - prev).rotate(dir);
                      });
        return points;
    }

public:
    void step(const Action &action) { game.step(action); }
    void reset() { game.reset(); }
    void switch_track(const size_t &idx) { game.switch_track(idx); }

    template <size_t N_FUTURE_POINTS> auto get_obs() const
    {
        return obs_t{ get_relative_vel(), get_relative_points<N_FUTURE_POINTS>() };
    }
    decltype(auto) get_game() const { return (game); }
};