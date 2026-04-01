#include "game/race.h"

#include <cmath>

constexpr auto DEFAULT_INDEX_FILE = "track_data/index";
constexpr auto DEFAULT_DATA_PATH = "track_data";

RaceState::RaceState()
    : last_lap_times{}, current_ticks{}, lap_count{}, stored_laps{}, next_vertex{ 1 }, vertex_count{}, started{}
{
}

void RaceState::reset(const size_t &vc)
{
    started = false;
    lap_count = 0;
    current_ticks = 0;
    stored_laps = 0;
    last_lap_times = {};
    next_vertex = 1;
    vertex_count = vc;
}

void RaceState::start()
{
    started = true;
}

void RaceState::advance()
{
    if (vertex_count == 0)
        return;

    next_vertex++;
    if (next_vertex >= vertex_count)
    {
        // All points visited => complete a lap
        next_vertex = 0;
        lap_count++;
        if (stored_laps < MAX_STORED_LAPS)
        {
            last_lap_times[stored_laps] = current_ticks;
            stored_laps++;
        }
        else
        {
            for (size_t i = 0; i < MAX_STORED_LAPS - 1; ++i)
                last_lap_times[i] = last_lap_times[i + 1];
            last_lap_times[MAX_STORED_LAPS - 1] = current_ticks;
        }
        current_ticks = 0;
    }
}

Gamestate::Gamestate() : maps{ DEFAULT_INDEX_FILE, DEFAULT_DATA_PATH }, race{}, car{} {}

template <class T> static bool segments_cross(T p1x, T p1y, T p2x, T p2y, T ax, T ay, T bx, T by)
{
    const auto d1x = p2x - p1x, d1y = p2y - p1y;
    const auto d2x = bx - ax, d2y = by - ay;
    const auto denom = d1x * d2y - d1y * d2x;
    if (std::abs(denom) < 1e-9)
        return false;
    const auto dx_ap = ax - p1x, dy_ap = ay - p1y;
    const auto u = (dx_ap * d2y - dy_ap * d2x) / denom;
    const auto t = (dx_ap * d1y - dy_ap * d1x) / denom;
    return u >= 0.0 && u <= 1.0 && t >= 0.0 && t <= 1.0;
}

static bool check_vertex_crossed(decltype(Car().get_pos()) prev, decltype(Car().get_pos()) curr,
                                 const TrackVertex &cp_vert)
{
    const auto mx = curr.x - prev.x;
    const auto my = curr.y - prev.y;
    const auto px = prev.x - mx;
    const auto py = prev.y - my;
    const auto cx = curr.x;
    const auto cy = curr.y;

    return segments_cross(px, py, cx, cy, cp_vert.left_grass_edge.x, cp_vert.left_grass_edge.y,
                          cp_vert.right_grass_edge.x, cp_vert.right_grass_edge.y);
}

void Gamestate::step(const Action &actions)
{
    if (!race.is_started() && actions.accelerate)
        race.start();

    const auto prev_pos = car.get_pos();

    car.step(actions, get_track());
    race.tick();

    if (!race.is_started())
        return;

    const auto &track = get_track();
    const auto &vertices = track.get_vertices();
    const auto &car_pos = car.get_pos();

    while (check_vertex_crossed(prev_pos, car_pos, vertices[race.get_next_vertex_idx()]))
    {
        race.advance();
    }
}

void Gamestate::reset()
{
    const auto &sp = get_track().get_start_pos();
    const auto &sd = get_track().get_start_dir();
    car = Car(sp.x, sp.y, sd);
    race.reset(get_track().get_vertices().size());
}

void Gamestate::switch_track(const size_t &map_idx)
{
    maps.set_track(map_idx);
    reset();
}
