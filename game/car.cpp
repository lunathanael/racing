#include "game/car.h"
#include "game/track.h"

#include <algorithm>
#include <cmath>

bool Car::is_drifting() const
{
    const auto forward_x = std::cos(dir);
    const auto forward_y = std::sin(dir);
    const auto right_x = -forward_y;
    const auto right_y = forward_x;
    const auto sideways_speed = vel.x * right_x + vel.y * right_y;
    return std::abs(sideways_speed) >= Car::STATIC_GRIP_THRESHOLD;
}

void Car::turn(bool is_clockwise, bool on_grass)
{
    const auto turn_speed = on_grass ? Car::TURN_SPEED_ON_GRASS : Car::TURN_SPEED;
    const auto delta = (is_clockwise ? turn_speed : -turn_speed);
    dir += delta * std::min<float_t>(1.0, get_speed2() / Car::ACCELERATION_SPEED);
    if (dir >= 2.0 * PI_V)
        dir -= 2.0 * PI_V;
    if (dir < 0.0)
        dir += 2.0 * PI_V;
}

void Car::accelerate()
{
    const auto forward_x = std::cos(dir);
    const auto forward_y = std::sin(dir);
    const auto right_x = -forward_y;
    const auto right_y = forward_x;

    auto forward_speed = vel.x * forward_x + vel.y * forward_y;
    auto sideways_speed = vel.x * right_x + vel.y * right_y;

    if (!is_drifting())
        forward_speed += Car::ACCELERATION_SPEED;
    else
        forward_speed += Car::ACCELERATION_SPEED * Car::STATIC_GRIP;

    forward_speed = std::min(forward_speed, Car::MAX_SPEED);

    vel.x = forward_x * forward_speed + right_x * sideways_speed;
    vel.y = forward_y * forward_speed + right_y * sideways_speed;
}

void Car::brake()
{
    const auto forward_x = std::cos(dir);
    const auto forward_y = std::sin(dir);
    const auto right_x = -forward_y;
    const auto right_y = forward_x;

    auto forward_speed = vel.x * forward_x + vel.y * forward_y;
    const auto sideways_speed = vel.x * right_x + vel.y * right_y;

    forward_speed -= Car::BREAKING_SPEED;
    forward_speed = std::max<float_t>(forward_speed, 0.0);

    vel.x = forward_x * forward_speed + right_x * sideways_speed;
    vel.y = forward_y * forward_speed + right_y * sideways_speed;
}

void Car::tick(bool force_drift)
{
    const auto forward_x = std::cos(dir);
    const auto forward_y = std::sin(dir);
    const auto right_x = -forward_y;
    const auto right_y = forward_x;

    auto forward_speed = vel.x * forward_x + vel.y * forward_y;
    auto sideways_speed = vel.x * right_x + vel.y * right_y;

    if (!force_drift && !is_drifting())
    {
        sideways_speed *= (1.0 - Car::STATIC_GRIP);
    }
    else
    {
        const auto abs_side = std::abs(sideways_speed);
        const auto sign = (sideways_speed > 0.0) ? 1.0 : -1.0;
        auto excess = abs_side - Car::STATIC_GRIP_THRESHOLD;
        excess *= (1.0 - Car::KINETIC_GRIP);
        sideways_speed = sign * (Car::STATIC_GRIP_THRESHOLD + excess);
    }

    forward_speed *= Car::DRAG;
    sideways_speed *= Car::DRAG;

    vel.x = forward_x * forward_speed + right_x * sideways_speed;
    vel.y = forward_y * forward_speed + right_y * sideways_speed;
}

void Car::move_with_collision(const Track &track)
{
    constexpr size_t n_steps = 1;
    for (size_t i = 0; i < n_steps; ++i)
    {
        pos.x += vel.x / n_steps;
        pos.y += vel.y / n_steps;

        auto px = pos.x, py = pos.y;
        auto vxf = vel.x, vyf = vel.y;

        if (track.resolve_wall_collision(px, py, vxf, vyf))
        {
            pos.x = px;
            pos.y = py;
            vel.x = vxf;
            vel.y = vyf;
        }
        else
        {
            Surface surf = track.get_surface(px, py);
            if (surf == Surface::GRASS)
            {
                vel.x *= GRASS_DRAG;
                vel.y *= GRASS_DRAG;
            }
        }
    }
}

void Car::step(const Action &actions, const Track &track)
{
    const Surface surf = track.get_surface(pos.x, pos.y);
    bool on_grass = (surf == Surface::GRASS);

    if (actions.left != actions.right)
    {
        if (actions.left)
            turn(false, on_grass);
        if (actions.right)
            turn(true, on_grass);
    }
    if (actions.accelerate)
        accelerate();
    if (actions.brake)
        brake();

    const bool force_drift = (actions.left != actions.right) && actions.brake;
    tick(force_drift);
    move_with_collision(track);
}
