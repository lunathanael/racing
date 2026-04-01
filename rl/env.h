#pragma once

#include "game/common.h"
#include "game/race.h"

#include <array>

class Env
{
public:
    static constexpr size_t N_FUTURE_POINTS = 120;
    using float_t = f64_t;
    struct obs_t
    {
        using points_t = std::array<vec<float_t>, N_FUTURE_POINTS>;
        vec<float_t> vel;
        points_t future_points;
    };

private:
    Gamestate game;

    vec<float_t> get_relative_vel() const;
    obs_t::points_t get_relative_points() const;

public:
    void step(const Action &action) { game.step(action); }
    void reset() { game.reset(); }
    void switch_track(const size_t &idx) { game.switch_track(idx); }
    obs_t get_obs() const;
    decltype(auto) get_game() const { return (game); }
};