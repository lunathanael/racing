#include "render/car.h"
#include "render/common.h"

#include <cmath>

// car rendering
void render_car(SDL_Renderer *renderer, float sx, float sy, double angle_rad)
{
    float half_w = BLOCK_SIZE * 2.0f; // slightly bigger
    float half_h = BLOCK_SIZE * 0.65f;
    float cos_a = std::cos((float)angle_rad);
    float sin_a = std::sin((float)angle_rad);

    auto rotate = [&](float lx, float ly, float &ox, float &oy)
    {
        ox = sx + lx * cos_a - ly * sin_a;
        oy = sy + lx * sin_a + ly * cos_a;
    };

    {
        SDL_FPoint pts[6];
        rotate(-half_w, -half_h, pts[0].x, pts[0].y);
        rotate(half_w * 0.85f, -half_h, pts[1].x, pts[1].y);
        rotate(half_w, -half_h * 0.4f, pts[2].x, pts[2].y);
        rotate(half_w, half_h * 0.4f, pts[3].x, pts[3].y);
        rotate(half_w * 0.85f, half_h, pts[4].x, pts[4].y);
        rotate(-half_w, half_h, pts[5].x, pts[5].y);

        SDL_FColor col = { 0.85f, 0.12f, 0.12f, 1.0f };
        SDL_Vertex verts[6];
        for (int i = 0; i < 6; i++)
        {
            verts[i].position = pts[i];
            verts[i].color = col;
            verts[i].tex_coord = { 0, 0 };
        }
        int idx[] = { 0, 1, 5, 1, 4, 5, 1, 2, 4, 2, 3, 4 };
        SDL_RenderGeometry(renderer, nullptr, verts, 6, idx, 12);
    }

    {
        float sw = half_h * 0.2f;
        SDL_FPoint sp[4];
        rotate(-half_w * 0.7f, -sw, sp[0].x, sp[0].y);
        rotate(half_w * 0.5f, -sw, sp[1].x, sp[1].y);
        rotate(half_w * 0.5f, sw, sp[2].x, sp[2].y);
        rotate(-half_w * 0.7f, sw, sp[3].x, sp[3].y);

        SDL_FColor wht = { 0.95f, 0.95f, 0.95f, 1.0f };
        SDL_Vertex sv[4];
        for (int i = 0; i < 4; i++)
        {
            sv[i].position = sp[i];
            sv[i].color = wht;
            sv[i].tex_coord = { 0, 0 };
        }
        int si[] = { 0, 1, 2, 0, 2, 3 };
        SDL_RenderGeometry(renderer, nullptr, sv, 4, si, 6);
    }

    {
        SDL_FColor wc = { 0.1f, 0.1f, 0.1f, 1.0f };
        float ww = half_w * 0.22f, wh = half_h * 0.35f;
        struct
        {
            float ox, oy;
        } wpos[] = {
            { -half_w * 0.6f, -half_h - wh * 0.3f },
            { -half_w * 0.6f, half_h - wh * 0.7f },
            { half_w * 0.4f, -half_h - wh * 0.3f },
            { half_w * 0.4f, half_h - wh * 0.7f },
        };
        for (auto &wp : wpos)
        {
            SDL_FPoint wpts[4];
            rotate(wp.ox, wp.oy, wpts[0].x, wpts[0].y);
            rotate(wp.ox + ww, wp.oy, wpts[1].x, wpts[1].y);
            rotate(wp.ox + ww, wp.oy + wh, wpts[2].x, wpts[2].y);
            rotate(wp.ox, wp.oy + wh, wpts[3].x, wpts[3].y);
            SDL_Vertex wv[4];
            for (int i = 0; i < 4; i++)
            {
                wv[i].position = wpts[i];
                wv[i].color = wc;
                wv[i].tex_coord = { 0, 0 };
            }
            int wi[] = { 0, 1, 2, 0, 2, 3 };
            SDL_RenderGeometry(renderer, nullptr, wv, 4, wi, 6);
        }
    }

    {
        float ws = half_w * 0.12f, woff = half_w * 0.25f;
        SDL_FPoint wp[4];
        rotate(woff, -half_h * 0.55f, wp[0].x, wp[0].y);
        rotate(woff + ws, -half_h * 0.4f, wp[1].x, wp[1].y);
        rotate(woff + ws, half_h * 0.4f, wp[2].x, wp[2].y);
        rotate(woff, half_h * 0.55f, wp[3].x, wp[3].y);

        SDL_FColor glass = { 0.45f, 0.65f, 0.85f, 0.9f };
        SDL_Vertex gv[4];
        for (int i = 0; i < 4; i++)
        {
            gv[i].position = wp[i];
            gv[i].color = glass;
            gv[i].tex_coord = { 0, 0 };
        }
        int gi[] = { 0, 1, 2, 0, 2, 3 };
        SDL_RenderGeometry(renderer, nullptr, gv, 4, gi, 6);
    }
}