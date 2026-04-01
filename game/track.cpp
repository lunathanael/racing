#include "game/track.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

template <class T>
auto point_to_segment_dist(const T &px, const T &py, const T &ax, const T &ay, const T &bx, const T &by, T &cx, T &cy)
{
    T dx = bx - ax, dy = by - ay;
    T len2 = dx * dx + dy * dy;
    T t = 0.0;
    if (len2 > 1e-6)
        t = std::clamp(((px - ax) * dx + (py - ay) * dy) / len2, T{ 0.0 }, T{ 1.0 });
    cx = ax + t * dx;
    cy = ay + t * dy;
    T ex = px - cx, ey = py - cy;
    return std::sqrt(ex * ex + ey * ey);
}

void Track::load_from_file(const std::string &path)
{
    std::ifstream file(path);
    assert(file.is_open());

    std::string token;

    // Parse: DIRECTION CW|CCW
    file >> token; // "DIRECTION"
    assert(token == "DIRECTION");
    file >> token; // "CW" or "CCW"
    clockwise = (token == "CW");

    // Parse: N_VERTICES <count>
    file >> token; // "N_VERTICES"
    assert(token == "N_VERTICES");
    size_t n;
    file >> n;

    vertices.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        file >> vertices[i].left_edge.x >> vertices[i].left_edge.y >> vertices[i].right_edge.x >>
            vertices[i].right_edge.y >> vertices[i].left_grass_edge.x >> vertices[i].left_grass_edge.y >>
            vertices[i].right_grass_edge.x >> vertices[i].right_grass_edge.y >> vertices[i].center.x >>
            vertices[i].center.y;
    }
}

f64_t Track::get_start_dir() const
{
    if (vertices.size() < 2)
        return 0.0;
    const auto &p0 = vertices[0].center;
    const auto &p1 = vertices[1].center;
    return std::atan2(p1.y - p0.y, p1.x - p0.x);
}

f64_t Track::distance_to_centerline(const f64_t &px, const f64_t &py, f64_t &closest_x, f64_t &closest_y) const
{
    f64_t best = 1e18;
    closest_x = px;
    closest_y = py;
    const size_t n = vertices.size();

    for (size_t i = 0; i < n; ++i)
    {
        size_t j = (i + 1) % n;
        const auto &ci = vertices[i].center;
        const auto &cj = vertices[j].center;
        f64_t cx, cy;
        f64_t d = point_to_segment_dist(px, py, ci.x, ci.y, cj.x, cj.y, cx, cy);
        if (d < best)
        {
            best = d;
            closest_x = cx;
            closest_y = cy;
        }
    }
    return best;
}

Surface Track::get_surface(const f64_t &px, const f64_t &py) const
{
    f64_t cx, cy;
    const f64_t d = distance_to_centerline(px, py, cx, cy);
    if (d <= ROAD_HALF_WIDTH)
        return Surface::ROAD;
    if (d <= ROAD_HALF_WIDTH + GRASS_WIDTH)
        return Surface::GRASS;
    return Surface::WALL;
}

bool Track::resolve_wall_collision(f64_t &px, f64_t &py, f64_t &vx, f64_t &vy) const
{
    f64_t cx, cy;
    f64_t d = distance_to_centerline(px, py, cx, cy);
    f64_t limit = ROAD_HALF_WIDTH + GRASS_WIDTH;
    if (d <= limit)
        return false;

    f64_t dx = px - cx, dy = py - cy;
    f64_t len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-6)
        return true;
    px = cx + dx / len * (limit - 1.0);
    py = cy + dy / len * (limit - 1.0);

    f64_t nx = dx / len, ny = dy / len;
    f64_t vn = vx * nx + vy * ny;
    if (vn > 0.0)
    {
        vx -= vn * nx;
        vy -= vn * ny;
    }
    vx *= 0.5;
    vy *= 0.5;
    return true;
}
