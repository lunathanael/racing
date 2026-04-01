#pragma once

#include "game/track.h"

#include <cmath>

struct Action
{
    bool accelerate{ false };
    bool brake{ false };
    bool left{ false };
    bool right{ false };

    void clear() { accelerate = brake = left = right = false; }
};

class Car
{
public:
    using float_t = f64_t;

    static constexpr float_t TURN_SPEED = 0.011 * (2.0 * PI_V);
    static constexpr float_t TURN_SPEED_ON_GRASS = 0.005 * (2.0 * PI_V);
    static constexpr float_t BREAKING_SPEED = 0.08;
    static constexpr float_t ACCELERATION_SPEED = 0.08967;
    static constexpr float_t MAX_SPEED = 35.0;
    static constexpr float_t DRAG = 0.9969;
    static constexpr float_t STATIC_GRIP = 0.7;
    static constexpr float_t KINETIC_GRIP = 0.08;
    static constexpr float_t STATIC_GRIP_THRESHOLD = 1.67;
    static constexpr float_t GRASS_DRAG = 0.99;

private:
    vec<float_t> pos;
    vec<float_t> vel;
    float_t dir;

public:
    Car(const decltype(pos.x) &pos_x, const decltype(pos.y) &pos_y, const decltype(dir) &init_dir)
        : pos{ pos_x, pos_y }, vel{}, dir{ init_dir }
    {
    }
    Car() : pos{}, vel{}, dir{} {}

    auto get_speed2() const { return (vel.x * vel.x) + (vel.y * vel.y); }
    auto get_speed() const { return std::sqrt(get_speed2()); }
    decltype(auto) get_pos() const { return (pos); }
    decltype(auto) get_vel() const { return (vel); }
    decltype(auto) get_dir() const { return (dir); };
    bool is_drifting() const;

    void turn(bool is_clockwise, bool on_grass = false);
    void accelerate();
    void brake();
    void tick(bool force_drift);
    void move_with_collision(const Track &track);
    void step(const Action &actions, const Track &track);
};
