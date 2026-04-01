#include "rl/env.h"

#include <algorithm>

vec<Env::float_t> Env::get_relative_vel() const
{
    const auto &car = game.get_car();
    const auto &vel = car.get_vel();
    const auto &dir = car.get_dir();

    return vel.rotate(dir);
}

Env::obs_t::points_t Env::get_relative_points() const
{
    Env::obs_t::points_t points;
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
                      if (idx == 0)
                      {
                          p = (vertices[next_point].center - pos).rotate(dir);
                          return;
                      }
                      const auto current_idx = (next_point + idx) % n_points;
                      const auto next_idx = (next_point + idx + 1) % n_points;
                      const auto &current = vertices[current_idx].center;
                      const auto &next = vertices[next_idx].center;
                      p = (next - current).rotate(dir);
                  });
    return points;
}

Env::obs_t Env::get_obs() const
{
    return { get_relative_vel(), get_relative_points() };
}