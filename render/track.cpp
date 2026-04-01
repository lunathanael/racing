#include "render/track.h"
#include "render/common.h"
#include <cmath>

void render_track(SDL_Renderer *renderer, const Track &track, float cam_x, float cam_y, bool debug_checkpoints)
{
    const auto &verts = track.get_vertices();
    const size_t n = verts.size();
    if (n < 2)
        return;

    const float margin = 300.0f;
    const float vleft = cam_x - SCREEN_W * 0.5f - margin;
    const float vright = cam_x + SCREEN_W * 0.5f + margin;
    const float vtop = cam_y - SCREEN_H * 0.5f - margin;
    const float vbot = cam_y + SCREEN_H * 0.5f + margin;

    auto W = [&](auto wx, auto wy) -> SDL_FPoint
    { return { (float)wx - cam_x + SCREEN_W * 0.5f, (float)wy - cam_y + SCREEN_H * 0.5f }; };

    const SDL_FColor grass_col = { 0.08f, 0.38f, 0.08f, 1.0f };
    const SDL_FColor road_col = { 0.20f, 0.20f, 0.22f, 1.0f };

    // Batch all quads into a single geometry call per layer
    std::vector<SDL_Vertex> vbuf;
    std::vector<int> ibuf;
    vbuf.reserve(n * 4);
    ibuf.reserve(n * 6);

    auto flush = [&](SDL_Renderer *r)
    {
        if (!ibuf.empty())
            SDL_RenderGeometry(r, nullptr, vbuf.data(), (int)vbuf.size(), ibuf.data(), (int)ibuf.size());
        vbuf.clear();
        ibuf.clear();
    };

    auto push_quad = [&](SDL_FPoint a0, SDL_FPoint a1, // left  i, i+1
                         SDL_FPoint b0, SDL_FPoint b1, // right i, i+1
                         SDL_FColor col)
    {
        int base = (int)vbuf.size();
        vbuf.push_back({ a0, col, { 0, 0 } });
        vbuf.push_back({ a1, col, { 0, 0 } });
        vbuf.push_back({ b1, col, { 0, 0 } });
        vbuf.push_back({ b0, col, { 0, 0 } });
        ibuf.push_back(base + 0);
        ibuf.push_back(base + 1);
        ibuf.push_back(base + 2);
        ibuf.push_back(base + 0);
        ibuf.push_back(base + 2);
        ibuf.push_back(base + 3);
    };

    auto visible = [vleft, vright, vtop, vbot](const auto &point)
    {
        const auto &cx = point.center.x;
        const auto &cy = point.center.y;
        return !(cx < vleft || cx > vright || cy < vtop || cy > vbot);
    };

    for (size_t i = 0; i < n; ++i)
    {
        const auto &prev = verts[(i + n - 1) % n];
        const auto &vi = verts[i];
        const auto &next = verts[(i + 1) % n];

        if (!visible(vi) && !visible(prev) && !visible(next))
            continue;

        const auto &vj = verts[(i + 1) % n];
        push_quad(W(vi.left_grass_edge.x, vi.left_grass_edge.y), W(vj.left_grass_edge.x, vj.left_grass_edge.y),
                  W(vi.right_grass_edge.x, vi.right_grass_edge.y), W(vj.right_grass_edge.x, vj.right_grass_edge.y),
                  grass_col);
    }
    flush(renderer);

    for (size_t i = 0; i < n; ++i)
    {
        const auto &prev = verts[(i + n - 1) % n];
        const auto &vi = verts[i];
        const auto &next = verts[(i + 1) % n];

        if (!visible(vi) && !visible(prev) && !visible(next))
            continue;

        const auto &vj = verts[(i + 1) % n];
        push_quad(W(vi.left_edge.x, vi.left_edge.y), W(vj.left_edge.x, vj.left_edge.y),
                  W(vi.right_edge.x, vi.right_edge.y), W(vj.right_edge.x, vj.right_edge.y), road_col);
    }
    flush(renderer);

    SDL_SetRenderDrawColorFloat(renderer, 0.72f, 0.72f, 0.75f, 1.0f);
    for (size_t i = 0; i < n; ++i)
    {
        const auto &prev = verts[(i + n - 1) % n];
        const auto &vi = verts[i];
        const auto &next = verts[(i + 1) % n];

        if (!visible(vi) && !visible(prev) && !visible(next))
            continue;

        const auto &vj = verts[(i + 1) % n];
        SDL_FPoint l0 = W(vi.left_edge.x, vi.left_edge.y);
        SDL_FPoint l1 = W(vj.left_edge.x, vj.left_edge.y);
        SDL_FPoint r0 = W(vi.right_edge.x, vi.right_edge.y);
        SDL_FPoint r1 = W(vj.right_edge.x, vj.right_edge.y);
        SDL_RenderLine(renderer, l0.x, l0.y, l1.x, l1.y);
        SDL_RenderLine(renderer, r0.x, r0.y, r1.x, r1.y);
    }

    if (debug_checkpoints)
    {
        SDL_SetRenderDrawColor(renderer, 255, 180, 0, 180);
        for (const auto &v : verts)
        {
            SDL_FPoint a = W(v.left_grass_edge.x, v.left_grass_edge.y);
            SDL_FPoint b = W(v.right_grass_edge.x, v.right_grass_edge.y);
            SDL_RenderLine(renderer, a.x, a.y, b.x, b.y);
        }
    }

    if (n > 0)
    {
        const auto &v0 = verts[0];
        SDL_FPoint a = W(v0.left_edge.x, v0.left_edge.y);
        SDL_FPoint b = W(v0.right_edge.x, v0.right_edge.y);
        float dx = b.x - a.x, dy = b.y - a.y;
        float len = std::sqrt(dx * dx + dy * dy);
        if (len < 1.0f)
            return;
        float nx = dx / len, ny = dy / len;
        float tx = dy / len, ty = -dx / len;
        const float sq = 5.0f;
        int num = (int)(len / sq);
        for (int r = 0; r < 2; ++r)
            for (int s = 0; s < num; ++s)
            {
                SDL_FColor c = ((s + r) % 2 == 0) ? SDL_FColor{ 1, 1, 1, 1 } : SDL_FColor{ 0, 0, 0, 1 };
                float ox = nx * (s * sq) + tx * (r * sq);
                float oy = ny * (s * sq) + ty * (r * sq);
                SDL_FPoint pts[4] = {
                    { a.x + ox, a.y + oy },
                    { a.x + ox + nx * sq, a.y + oy + ny * sq },
                    { a.x + ox + nx * sq + tx * sq, a.y + oy + ny * sq + ty * sq },
                    { a.x + ox + tx * sq, a.y + oy + ty * sq },
                };
                SDL_Vertex sv[4];
                for (int k = 0; k < 4; k++)
                    sv[k] = { pts[k], c, { 0, 0 } };
                int qi[] = { 0, 1, 2, 0, 2, 3 };
                SDL_RenderGeometry(renderer, nullptr, sv, 4, qi, 6);
            }
    }
}