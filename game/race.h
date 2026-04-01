#pragma once

#include "game/car.h"
#include "game/maps.h"

#include <array>
#include <cstdint>
class RaceState
{
public:
    using ticks_t = uint64_t;
    static constexpr size_t MAX_STORED_LAPS = 3;

private:
    std::array<ticks_t, MAX_STORED_LAPS> last_lap_times;
    ticks_t current_ticks;

    size_t lap_count;
    size_t stored_laps;
    size_t next_vertex;
    size_t vertex_count;

    bool started;

public:
    RaceState();
    void reset(const size_t &vertex_count);

    decltype(auto) get_lap_time() const { return (current_ticks); }
    decltype(auto) get_lap_count() const { return (lap_count); }
    decltype(auto) get_last_lap_times() const { return (last_lap_times); }
    decltype(auto) get_stored_lap_count() const { return (stored_laps); }

    bool is_started() const { return started; }
    void start();

    decltype(auto) get_next_vertex_idx() const { return (next_vertex); }
    decltype(auto) get_vertex_count() const { return (vertex_count); }

    void tick()
    {
        if (is_started())
            ++current_ticks;
    }
    void advance();
};

class Gamestate
{
    Maps maps;
    RaceState race;
    Car car;

public:
    Gamestate();

    void step(const Action &actions);
    void reset();
    void switch_track(const size_t &map_idx);

    decltype(auto) get_car() const { return (car); }
    decltype(auto) get_track() const { return maps.get_current_track(); }
    decltype(auto) get_race() const { return (race); }
    decltype(auto) get_maps() const { return (maps); }
    decltype(auto) get_current_map_idx() const { return maps.get_current_idx(); }
};
