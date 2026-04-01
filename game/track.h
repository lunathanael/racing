#pragma once

#include "game/common.h"

#include <cstddef>
#include <string>
#include <vector>

enum class Surface
{
    ROAD,
    GRASS,
    WALL
};

using TrackPoint = vec<f64_t>;

struct TrackVertex
{
    TrackPoint left_edge;
    TrackPoint right_edge;
    TrackPoint left_grass_edge;
    TrackPoint right_grass_edge;
    TrackPoint center;
};

class Track
{
    std::vector<TrackVertex> vertices;
    bool clockwise;

public:
    static constexpr f64_t ROAD_HALF_WIDTH = 40.5;
    static constexpr f64_t GRASS_WIDTH = 40.0;

    void load_from_file(const std::string &path);

    decltype(auto) get_point_count() const { return vertices.size(); }
    decltype(auto) get_vertices() const { return (vertices); }

    decltype(auto) get_centerpoint(const size_t &i) const { return vertices[i].center; }
    decltype(auto) get_start_pos() const
    {
        assert(!vertices.empty());
        return vertices[0].center;
    }

    f64_t get_start_dir() const;

    decltype(auto) is_clockwise() const { return (clockwise); }

    Surface get_surface(const f64_t &px, const f64_t &py) const;
    f64_t distance_to_centerline(const f64_t &px, const f64_t &py, f64_t &closest_x, f64_t &closest_y) const;
    bool resolve_wall_collision(f64_t &px, f64_t &py, f64_t &vx, f64_t &vy) const;
};
