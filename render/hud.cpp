#include "render/hud.h"
#include "render/common.h"

#include <SDL3/SDL.h>

#include <cstdio>
#include <format>

// 7-segment digit rendering

static const bool SEGMENTS[10][7] = {
    { true, true, true, true, true, true, false },   { false, true, true, false, false, false, false },
    { true, true, false, true, true, false, true },  { true, true, true, true, false, false, true },
    { false, true, true, false, false, true, true }, { true, false, true, true, false, true, true },
    { true, false, true, true, true, true, true },   { true, true, true, false, false, false, false },
    { true, true, true, true, true, true, true },    { true, true, true, true, false, true, true },
};

static void draw_digit(SDL_Renderer *r, int d, float x, float y, float w, float h)
{
    if (d < 0 || d > 9)
        return;
    const bool *s = SEGMENTS[d];
    float t = h * 0.08f;
    float hh = h * 0.5f;
    if (s[0])
    {
        SDL_FRect rc = { x, y, w, t };
        SDL_RenderFillRect(r, &rc);
    }
    if (s[3])
    {
        SDL_FRect rc = { x, y + h - t, w, t };
        SDL_RenderFillRect(r, &rc);
    }
    if (s[6])
    {
        SDL_FRect rc = { x, y + hh - t * 0.5f, w, t };
        SDL_RenderFillRect(r, &rc);
    }
    if (s[1])
    {
        SDL_FRect rc = { x + w - t, y, t, hh };
        SDL_RenderFillRect(r, &rc);
    }
    if (s[2])
    {
        SDL_FRect rc = { x + w - t, y + hh, t, hh };
        SDL_RenderFillRect(r, &rc);
    }
    if (s[4])
    {
        SDL_FRect rc = { x, y + hh, t, hh };
        SDL_RenderFillRect(r, &rc);
    }
    if (s[5])
    {
        SDL_FRect rc = { x, y, t, hh };
        SDL_RenderFillRect(r, &rc);
    }
}

void draw_hud_string(SDL_Renderer *renderer, const char *str, float x, float y, float cw, float ch, float sp)
{
    float cx = x;
    for (const char *p = str; *p; ++p)
    {
        if (*p >= '0' && *p <= '9')
        {
            draw_digit(renderer, *p - '0', cx, y, cw, ch);
            cx += cw + sp;
        }
        else if (*p == '.')
        {
            float s = ch * 0.12f;
            SDL_FRect rc = { cx + cw * 0.15f, y + ch - s * 1.5f, s, s };
            SDL_RenderFillRect(renderer, &rc);
            cx += cw * 0.4f + sp;
        }
        else if (*p == ':')
        {
            float s = ch * 0.12f;
            SDL_FRect r1 = { cx + cw * 0.15f, y + ch * 0.28f, s, s };
            SDL_FRect r2 = { cx + cw * 0.15f, y + ch * 0.65f, s, s };
            SDL_RenderFillRect(renderer, &r1);
            SDL_RenderFillRect(renderer, &r2);
            cx += cw * 0.4f + sp;
        }
        else if (*p == '/')
        {
            float thickness = 2.0f;
            for (float i = 0; i < thickness; i += 0.5f)
            {
                SDL_RenderLine(renderer, cx + cw * 0.75f + i, y + ch * 0.05f, cx + cw * 0.25f + i, y + ch * 0.95f);
            }
            cx += cw * 1.0f + sp;
        }
        else
        {
            cx += cw * 0.5f + sp;
        }
    }
}

// HUD rendering

void render_speedometer(SDL_Renderer *renderer, float speed)
{
    float display_speed = speed * 10.0f;
    if (display_speed > 999.9f)
        display_speed = 999.9f;

    char buf[8];
    std::snprintf(buf, sizeof(buf), "%05.1f", display_speed);

    float box_w = 140, box_h = 50;
    float box_x = 15, box_y = SCREEN_H - box_h - 15;

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 10, 10, 15, 200);
    SDL_FRect bg = { box_x, box_y, box_w, box_h };
    SDL_RenderFillRect(renderer, &bg);
    SDL_SetRenderDrawColor(renderer, 80, 80, 90, 255);
    SDL_RenderRect(renderer, &bg);

    SDL_SetRenderDrawColor(renderer, 0, 255, 100, 255);
    draw_hud_string(renderer, buf, box_x + 10, box_y + 8, 20, 34, 3);
}

void render_lap_info(SDL_Renderer *renderer, const RaceState &race, uint64_t now_ms, bool debug_checkpoints)
{
    float line_h = 26;
    int lines = 1 + static_cast<int>(race.get_stored_lap_count());
    if (race.get_vertex_count() > 0)
        lines++;
    float box_w = 190, box_h = 10 + lines * line_h + 5;
    float box_x = SCREEN_W - box_w - 15, box_y = 15;

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 10, 10, 15, 200);
    SDL_FRect bg = { box_x, box_y, box_w, box_h };
    SDL_RenderFillRect(renderer, &bg);
    SDL_SetRenderDrawColor(renderer, 80, 80, 90, 255);
    SDL_RenderRect(renderer, &bg);

    float tx = box_x + 8, ty = box_y + 8;
    float cw = 14, ch = 22, sp = 2;

    const auto cur_ticks = race.get_lap_time();
    auto cur = static_cast<double>(cur_ticks) / TICKS_PER_SECOND;
    int mins = (int)(cur / 60.0);
    float secs = static_cast<float>(cur - mins * 60);
    char cur_buf[32];
    std::snprintf(cur_buf, sizeof(cur_buf), "%d:%05.2f", mins, secs);

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    draw_hud_string(renderer, cur_buf, tx, ty, cw, ch, sp);
    ty += line_h;

    if (debug_checkpoints && race.get_vertex_count() > 0)
    {
        const auto checkpoint_str = std::format("{}/{}", race.get_next_vertex_idx(), race.get_vertex_count());
        SDL_SetRenderDrawColor(renderer, 255, 180, 0, 255);
        draw_hud_string(renderer, checkpoint_str.c_str(), tx, ty, cw * 0.7f, ch * 0.7f, sp);
        ty += line_h * 0.8f;
    }

    const auto &laps = race.get_last_lap_times();
    for (size_t i = 0; i < race.get_stored_lap_count(); ++i)
    {
        int lm = (int)(static_cast<double>(laps[i]) / 60.0);
        float ls = static_cast<float>(laps[i] - lm * 60);
        char lap_buf[32];
        std::snprintf(lap_buf, sizeof(lap_buf), "%d:%05.2f", lm, ls);
        SDL_SetRenderDrawColor(renderer, 140, 140, 255, 255);
        draw_hud_string(renderer, lap_buf, tx, ty, cw * 0.85f, ch * 0.85f, sp);
        ty += line_h;
    }
}

void render_minimap(SDL_Renderer *renderer, const Track &track, const Car &car)
{
    const size_t n = track.get_point_count();
    if (n == 0)
        return;

    double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
    for (size_t i = 0; i < n; ++i)
    {
        TrackPoint p = track.get_centerpoint(i);
        if (p.x < minx)
            minx = p.x;
        if (p.x > maxx)
            maxx = p.x;
        if (p.y < miny)
            miny = p.y;
        if (p.y > maxy)
            maxy = p.y;
    }

    float map_w = 360, map_h = 280;
    float map_x = 15, map_y = 15;
    float pad = 15;

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 10, 10, 15, 180);
    SDL_FRect bg = { map_x, map_y, map_w, map_h };
    SDL_RenderFillRect(renderer, &bg);
    SDL_SetRenderDrawColor(renderer, 80, 80, 90, 255);
    SDL_RenderRect(renderer, &bg);

    float scale_x = (map_w - 2 * pad) / static_cast<float>(maxx - minx);
    float scale_y = (map_h - 2 * pad) / static_cast<float>(maxy - miny);
    float scale = std::min(scale_x, scale_y);

    float off_x = map_x + pad + ((map_w - 2 * pad) - static_cast<float>(maxx - minx) * scale) * 0.5f;
    float off_y = map_y + pad + ((map_h - 2 * pad) - static_cast<float>(maxy - miny) * scale) * 0.5f;

    auto to_map = [&](auto wx, auto wy) -> SDL_FPoint
    { return { off_x + static_cast<float>(wx - minx) * scale, off_y + static_cast<float>(wy - miny) * scale }; };

    SDL_SetRenderDrawColor(renderer, 160, 160, 170, 255);
    for (size_t i = 0; i < n; ++i)
    {
        size_t j = (i + 1) % n;
        TrackPoint pi = track.get_centerpoint(i);
        TrackPoint pj = track.get_centerpoint(j);
        SDL_FPoint a = to_map(pi.x, pi.y);
        SDL_FPoint b = to_map(pj.x, pj.y);
        SDL_RenderLine(renderer, a.x, a.y, b.x, b.y);
    }

    SDL_FPoint cp = to_map(car.get_pos().x, car.get_pos().y);
    SDL_SetRenderDrawColor(renderer, 255, 50, 50, 255);
    SDL_FRect dot = { cp.x - 3, cp.y - 3, 6, 6 };
    SDL_RenderFillRect(renderer, &dot);
}

void render_inputs(SDL_Renderer *renderer, const Action &actions)
{
    const float PAD = 18.0f;
    const float SZ = 52.0f;
    const float GAP = 6.0f;
    const float RADIUS = 8.0f;

    // Bottom-right anchor
    const float right = SCREEN_W - PAD;
    const float bottom = SCREEN_H - PAD;

    struct Key
    {
        const char *label;
        float cx, cy;
        bool pressed;
    } keys[] = {
        { "W", right - SZ - GAP - SZ * 0.5f, bottom - SZ - GAP - SZ * 0.5f, actions.accelerate },
        { "A", right - SZ * 2 - GAP * 2 - SZ * 0.5f, bottom - SZ * 0.5f, actions.left },
        { "S", right - SZ - GAP - SZ * 0.5f, bottom - SZ * 0.5f, actions.brake },
        { "D", right - SZ * 0.5f, bottom - SZ * 0.5f, actions.right },
    };

    for (auto &k : keys)
    {
        float x = k.cx - SZ * 0.5f;
        float y = k.cy - SZ * 0.5f;

        // Background
        SDL_FColor bg = k.pressed ? SDL_FColor{ 0.95f, 0.85f, 0.15f, 1.00f }  // bright yellow when held
                                  : SDL_FColor{ 0.15f, 0.15f, 0.15f, 0.55f }; // dim dark when idle

        auto filled_rect = [&](float rx, float ry, float rw, float rh, SDL_FColor c)
        {
            SDL_Vertex v[4] = {
                { { rx, ry }, c, { 0, 0 } },
                { { rx + rw, ry }, c, { 0, 0 } },
                { { rx + rw, ry + rh }, c, { 0, 0 } },
                { { rx, ry + rh }, c, { 0, 0 } },
            };
            int idx[] = { 0, 1, 2, 0, 2, 3 };
            SDL_RenderGeometry(renderer, nullptr, v, 4, idx, 6);
        };

        filled_rect(x + RADIUS, y, SZ - RADIUS * 2, SZ, bg);
        filled_rect(x, y + RADIUS, SZ, SZ - RADIUS * 2, bg);

        SDL_FColor border = k.pressed ? SDL_FColor{ 1.0f, 1.0f, 0.4f, 1.0f } : SDL_FColor{ 0.6f, 0.6f, 0.6f, 0.7f };
        SDL_SetRenderDrawColorFloat(renderer, border.r, border.g, border.b, border.a);
        SDL_RenderLine(renderer, x + RADIUS, y, x + SZ - RADIUS, y);
        SDL_RenderLine(renderer, x + RADIUS, y + SZ, x + SZ - RADIUS, y + SZ);
        SDL_RenderLine(renderer, x, y + RADIUS, x, y + SZ - RADIUS);
        SDL_RenderLine(renderer, x + SZ, y + RADIUS, x + SZ, y + SZ - RADIUS);

        SDL_SetRenderDrawColorFloat(renderer, k.pressed ? 0.1f : 0.9f, k.pressed ? 0.1f : 0.9f, k.pressed ? 0.1f : 0.9f,
                                    1.0f);
        SDL_RenderDebugTextFormat(renderer, k.cx - 4.0f, k.cy - 4.0f, "%s", k.label);
    }
}

void render_map_selector(SDL_Renderer *renderer, const Gamestate &game)
{
    const auto &maps = game.get_maps();
    auto total = maps.size();

    if (total > 0)
    {
        size_t sidx = game.get_current_map_idx();
        const TrackMetadata &info = maps.get_metadata(sidx);

        SDL_SetRenderDrawColor(renderer, 100, 255, 100, 255);
        SDL_RenderDebugText(renderer, SCREEN_W / 2 - info.name.length() * 4, 120, info.name.c_str());

        const float preview_w = 800;
        const float preview_h = 600;
        const float ox = SCREEN_W / 2 - preview_w / 2;
        const float oy = SCREEN_H / 2 - preview_h / 2;

        SDL_FRect bg = { ox, oy, preview_w, preview_h };
        SDL_SetRenderDrawColor(renderer, 15, 15, 20, 255);
        SDL_RenderFillRect(renderer, &bg);
        SDL_SetRenderDrawColor(renderer, 100, 100, 120, 255);
        SDL_RenderRect(renderer, &bg);

        const auto &track = game.get_track();
        size_t n_pts = track.get_point_count();
        if (n_pts > 0)
        {
            f64_t minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
            for (size_t i = 0; i < n_pts; ++i)
            {
                TrackPoint p = track.get_centerpoint(i);
                if (p.x < minx)
                    minx = p.x;
                if (p.x > maxx)
                    maxx = p.x;
                if (p.y < miny)
                    miny = p.y;
                if (p.y > maxy)
                    maxy = p.y;
            }
            float scale_x = (preview_w - 40) / static_cast<float>(maxx - minx);
            float scale_y = (preview_h - 40) / static_cast<float>(maxy - miny);
            float scale = std::min(scale_x, scale_y);
            float cx = ox + 20 + ((preview_w - 40) - static_cast<float>(maxx - minx) * scale) * 0.5f;
            float cy = oy + 20 + ((preview_h - 40) - static_cast<float>(maxy - miny) * scale) * 0.5f;

            SDL_SetRenderDrawColor(renderer, 200, 200, 220, 255);
            for (size_t i = 0; i < n_pts; ++i)
            {
                size_t j = (i + 1) % n_pts;
                TrackPoint pi = track.get_centerpoint(i);
                TrackPoint pj = track.get_centerpoint(j);
                float px = cx + static_cast<float>(pi.x - minx) * scale;
                float py = cy + static_cast<float>(pi.y - miny) * scale;
                float nx = cx + static_cast<float>(pj.x - minx) * scale;
                float ny = cy + static_cast<float>(pj.y - miny) * scale;
                SDL_RenderLine(renderer, px, py, nx, ny);
                SDL_RenderLine(renderer, px + 1, py, nx + 1, ny);
                SDL_RenderLine(renderer, px - 1, py, nx - 1, ny);
                SDL_RenderLine(renderer, px, py + 1, nx, ny + 1);
                SDL_RenderLine(renderer, px, py - 1, nx, ny - 1);
            }
        }

        SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
        char idx_str[16];
        std::snprintf(idx_str, sizeof(idx_str), "%d/%d", static_cast<int>(sidx + 1), static_cast<int>(total));
        SDL_RenderDebugText(renderer, SCREEN_W / 2 - 20, oy + preview_h + 20, idx_str);
        SDL_RenderDebugText(renderer, SCREEN_W / 2 - 80, oy + preview_h + 60, "< LEFT   RIGHT >");
    }
}