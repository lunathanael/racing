#include "render/drift.h"
#include "game/common.h"
#include "render/common.h"

#include <algorithm>
#include <cmath>
#include <vector>

// drift particles
struct DriftParticle
{
    float x, y;
    float vx, vy;
    float life; // [0,1]
    float max_life;
};

static std::vector<DriftParticle> drift_particles;

void spawn_drift_particles(float car_x, float car_y, float car_dir, float speed)
{
    float cos_a = std::cos(car_dir);
    float sin_a = std::sin(car_dir);
    // spawn from rear wheels
    for (const int side : { -1, 1 })
    {
        float wx = car_x - cos_a * BLOCK_SIZE * 1.0f - (-sin_a) * side * BLOCK_SIZE * 0.7f;
        float wy = car_y - sin_a * BLOCK_SIZE * 1.0f - (cos_a)*side * BLOCK_SIZE * 0.7f;
        DriftParticle p;
        p.x = wx;
        p.y = wy;
        p.vx = (float)(rand() % 100 - 50) * 0.01f;
        p.vy = (float)(rand() % 100 - 50) * 0.01f;
        p.max_life = 0.4f + (float)(rand() % 30) * 0.01f;
        p.life = p.max_life;
        drift_particles.push_back(p);
    }
}

void update_drift_particles()
{
    for (auto &p : drift_particles)
    {
        p.x += p.vx;
        p.y += p.vy;
        p.life -= static_cast<float>(SECONDS_PER_TICK);
    }
    // remove dead particles
    drift_particles.erase(std::remove_if(drift_particles.begin(), drift_particles.end(),
                                         [](const DriftParticle &p) { return p.life <= 0; }),
                          drift_particles.end());
}

void render_drift_particles(SDL_Renderer *renderer, float cam_x, float cam_y)
{
    for (const auto &p : drift_particles)
    {
        float alpha = p.life / p.max_life;
        float sx = p.x - cam_x + SCREEN_W * 0.5f;
        float sy = p.y - cam_y + SCREEN_H * 0.5f;
        if (sx < -20 || sx > SCREEN_W + 20 || sy < -20 || sy > SCREEN_H + 20)
            continue;
        int a = (int)(alpha * 150);
        float size = 3.0f + (1.0f - alpha) * 4.0f;
        SDL_SetRenderDrawColor(renderer, 200, 200, 200, a);
        SDL_FRect r = { sx - size * 0.5f, sy - size * 0.5f, size, size };
        SDL_RenderFillRect(renderer, &r);
    }
}
