#!/usr/bin/env python3
import os
import sys
import math
import json

import cv2
import numpy as np
from skimage.morphology import skeletonize


MAX_DIST_FOR_NEIGHBOR = 20


def WARN(message):
    print(f"\033[33mWARNING: {message}\033[0m")


def get_nbrs(p, pts):
    x, y = p
    max_dist = MAX_DIST_FOR_NEIGHBOR
    for i in range(1, max_dist):
        ls = [
            (x + dx, y + dy)
            for dx in (-i, 0, i)
            for dy in (-i, 0, i)
            if (dx, dy) != (0, 0) and (x + dx, y + dy) in pts
        ]
        if ls:
            return ls
    return []


def largest_component(pts: set) -> set:
    remaining = set(pts)
    best = set()
    while remaining:
        seed = next(iter(remaining))
        comp = set()
        stack = [seed]
        while stack:
            p = stack.pop()
            if p in comp:
                continue
            comp.add(p)
            stack.extend(n for n in get_nbrs(p, pts) if n not in comp)
        remaining -= comp
        if len(comp) > len(best):
            best = comp
    return best


def prune_branches(pts: set, min_branch: int = 15) -> set:
    pts = set(pts)
    for _ in range(min_branch * 3):
        dead = [p for p in pts if len(get_nbrs(p, pts)) <= 1]
        if not dead:
            break
        pts -= set(dead)
    return pts


def walk_loop(pts: set, start: tuple) -> list:
    path = [start]
    visited = {start}
    current = start

    while True:
        nbrs = [n for n in get_nbrs(current, pts) if n not in visited]

        if nbrs:
            if len(nbrs) == 1:
                nxt = nbrs[0]
            else:

                def reachable(seed):
                    seen = {seed}
                    stack = [seed]
                    while stack:
                        p = stack.pop()
                        for nb in get_nbrs(p, pts):
                            if nb not in visited and nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
                    return len(seen)

                nxt = max(nbrs, key=reachable)
        else:

            dist = math.hypot(current[0] - start[0], current[1] - start[1])
            if dist <= 51 and len(path) > 20:
                break

            unvisited = pts - visited
            if not unvisited:
                break
            nxt = min(
                unvisited,
                key=lambda p: (p[0] - current[0]) ** 2 + (p[1] - current[1]) ** 2,
            )

        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path


def iron_out_start_bump(path, radius=35):
    n = len(path)
    if n < radius * 4:
        return path

    p_pre = path[-radius]
    p_post = path[radius]

    patch = []
    for i in range(-radius, radius):
        t = (i + radius) / (2 * radius)
        x = p_pre[0] * (1 - t) + p_post[0] * t
        y = p_pre[1] * (1 - t) + p_post[1] * t
        patch.append((x, y))

    return patch[radius:] + path[radius : n - radius] + patch[:radius]


def resample_uniform(centerline: np.ndarray, num_points: int = 10000) -> np.ndarray:
    """Evenly spaces points, perfectly bridging the start-line gap."""
    deltas = np.diff(centerline, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]
    if total < 1e-9:
        return centerline
    targets = np.linspace(0, total, num_points, endpoint=False)
    new_x = np.interp(targets, cum, centerline[:, 0])
    new_y = np.interp(targets, cum, centerline[:, 1])
    return np.column_stack((new_x, new_y))


def find_red_arrow(img: np.ndarray):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 70), (20, 255, 255)),
        cv2.inRange(hsv, (160, 70, 70), (180, 255, 255)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > (h * w) * 0.005:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area < 1 or area / hull_area < 0.70:
            continue
        candidates.append((area, c))

    if not candidates:
        return None, None
    _, arrow_c = max(
        candidates, key=lambda ac: ac[0] / cv2.contourArea(cv2.convexHull(ac[1]))
    )
    M = cv2.moments(arrow_c)
    if M["m00"] == 0:
        return None, None
    centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)

    pts2 = arrow_c[:, 0, :].astype(np.float32)
    _, _, vt = np.linalg.svd(pts2 - centroid, full_matrices=False)
    axis = vt[0]
    proj = (pts2 - centroid) @ axis
    direction = pts2[proj.argmax()] - pts2[proj.argmin()]
    norm = np.linalg.norm(direction)

    if norm < 1e-9:
        return None, None
    return tuple(centroid.astype(int)), direction / norm


def find_start_pixel(img, track_pts, arrow_centroid):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    acc = np.zeros((h, w), np.float32)

    for s in range(3, 22, 2):
        d = np.abs(gray[s:, s:] - gray[:-s, s:] - gray[s:, :-s] + gray[:-s, :-s])
        full = np.zeros((h, w), np.float32)
        full[s:, s:] = d
        mx = full.max()
        if mx > 0:
            full /= mx
        k = s * 4 + 1 | 1
        full = cv2.GaussianBlur(full, (k, k), s * 1.5)
        acc += full

    pts_arr = np.array(list(track_pts), dtype=np.int32)

    skel_img = np.ones((h, w), dtype=np.uint8) * 255
    skel_img[pts_arr[:, 1], pts_arr[:, 0]] = 0
    dist_to_track = cv2.distanceTransform(skel_img, cv2.DIST_L2, 3)
    valid_mask = (dist_to_track <= 120).astype(np.float32)

    if arrow_centroid is not None:
        arrow_img = np.ones((h, w), dtype=np.uint8) * 255
        arrow_img[arrow_centroid[1], arrow_centroid[0]] = 0
        dist_to_arrow = cv2.distanceTransform(arrow_img, cv2.DIST_L2, 3)
        valid_mask *= (dist_to_arrow <= (w * 0.25)).astype(np.float32)

    acc *= valid_mask

    _, max_val, _, max_loc = cv2.minMaxLoc(acc)

    best = None
    if max_val > 0.0:
        best = max_loc

    if best is None:
        if arrow_centroid is not None:
            WARN(
                "[start] accumulator found no valid texture near arrow, falling back to arrow"
            )
            best = arrow_centroid
        else:
            WARN("[start] no markers found, falling back to bottommost point")
            best = max(track_pts, key=lambda p: p[1])

    d = pts_arr - np.array(best)
    idx = np.einsum("ij,ij->i", d, d).argmin()
    snapped = tuple(pts_arr[idx])
    return snapped


def extract_skeleton(img: np.ndarray) -> set | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is not None:
        for i, c in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                if cv2.contourArea(c) < 5000:
                    cv2.drawContours(thresh, [c], -1, 255, -1)

    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist < 3:
        return None

    core_mask = (dist >= max_dist * 0.35).astype(np.uint8) * 255
    skel = skeletonize(core_mask > 0)

    ys, xs = np.where(skel)
    if len(xs) < 20:
        return None

    pts = set(zip(xs.tolist(), ys.tolist()))
    pts = largest_component(pts)
    pts = prune_branches(pts, min_branch=20)

    return pts if len(pts) >= 20 else None


def generate_edges(cl, road_hw, grass_w):
    from scipy.ndimage import gaussian_filter1d

    n = len(cl)

    def _s1(a, sig):
        return gaussian_filter1d(a, sig, mode="wrap")

    # Smooth only for normal estimation
    def _s2(a, sig):
        return np.stack(
            [
                gaussian_filter1d(a[:, 0], sig, mode="wrap"),
                gaussian_filter1d(a[:, 1], sig, mode="wrap"),
            ],
            axis=1,
        )

    cs = _s2(cl, 3)
    tang = np.roll(cs, -1, 0) - np.roll(cs, 1, 0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

    pad = int(max(road_hw, grass_w) * 4 + 20)
    mn = cl.min(0) - pad
    W = int(cl[:, 0].max() - cl[:, 0].min()) + pad * 2 + 4
    H = int(cl[:, 1].max() - cl[:, 1].min()) + pad * 2 + 4

    def make_mask(radius):
        mask = np.zeros((H, W), np.uint8)
        px = (cl - mn).astype(np.int32)
        for i in range(n):
            cv2.line(
                mask, tuple(px[i]), tuple(px[(i + 1) % n]), 255, max(1, int(radius * 2))
            )
        blur = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.5)
        return (blur > 100).astype(np.uint8) * 255

    road_mask = make_mask(road_hw)
    grass_mask = make_mask(road_hw + grass_w)

    def measure_width(mask, nominal_hw):
        S = max(40, int(nominal_hw * 3))
        steps = np.linspace(0.5, nominal_hw + 3, S, dtype=np.float32)

        l_rays = cl[:, None, :] + norm[:, None, :] * steps[None, :, None]
        r_rays = cl[:, None, :] - norm[:, None, :] * steps[None, :, None]

        def sample(rays):
            px = (rays - mn).astype(np.int32)
            px[..., 0] = np.clip(px[..., 0], 0, W - 1)
            px[..., 1] = np.clip(px[..., 1], 0, H - 1)
            return mask[px[..., 1], px[..., 0]] > 128

        def last_step(inside):
            rev = inside[:, ::-1]
            first_out = np.argmax(~rev, axis=1)
            last_in = np.where(inside.all(1), S - 1, S - 1 - first_out)
            return np.where(inside.any(1), steps[last_in], nominal_hw)

        return last_step(sample(l_rays)), last_step(sample(r_rays))

    l_road_w, r_road_w = measure_width(road_mask, road_hw)
    l_grass_w, r_grass_w = measure_width(grass_mask, road_hw + grass_w)

    l_road_w = _s1(l_road_w, 1.5)
    r_road_w = _s1(r_road_w, 1.5)
    l_grass_w = _s1(l_grass_w, 2.0)
    r_grass_w = _s1(r_grass_w, 2.0)

    lr = cl + norm * l_road_w[:, None]
    rr = cl - norm * r_road_w[:, None]
    lg = cl + norm * l_grass_w[:, None]
    rg = cl - norm * r_grass_w[:, None]

    cw = (
        float(
            np.sum(cl[:, 0] * np.roll(cl[:, 1], -1) - np.roll(cl[:, 0], -1) * cl[:, 1])
        )
        > 0
    )
    if cw:
        lr, rr = rr, lr
        lg, rg = rg, lg

    max_perp = 0.0
    worst_i = 0
    for i in range(n):
        nx, ny = norm[i]
        for name, pt in [("lr", lr[i]), ("rr", rr[i]), ("lg", lg[i]), ("rg", rg[i])]:
            dx = pt[0] - cl[i][0]
            dy = pt[1] - cl[i][1]
            perp = abs(dx * ny - dy * nx)
            if perp > max_perp:
                max_perp = perp
                worst_i = i
                worst_name = name

    if max_perp > 1.0:
        WARN(
            f"Edges NOT collinear: max perpendicular deviation = {max_perp:.3f} "
            f"at vertex {worst_i} ({worst_name}). "
            f"cl={cl[worst_i]}, norm={norm[worst_i]}"
        )

    return np.array(lr), np.array(rr), np.array(lg), np.array(rg), cw


def resample_adaptive(
    cl: np.ndarray,
    min_dist: float = 2.0,
    max_dist: float = 20.0,
    max_angle_deg: float = 3.0,
) -> np.ndarray:

    n = len(cl)
    if n < 3:
        return cl.copy()

    out = [cl[0], cl[1]]

    for i in range(2, n):
        A = out[-2]
        B = out[-1]
        C = cl[i]

        dist_from_last = math.hypot(C[0] - B[0], C[1] - B[1])

        if dist_from_last < min_dist:
            continue

        v1x = B[0] - A[0]
        v1y = B[1] - A[1]
        v2x = C[0] - B[0]
        v2y = C[1] - B[1]

        dot = v1x * v2x + v1y * v2y
        cross = v1x * v2y - v1y * v2x

        if abs(cross) < 1:
            continue

        da = abs(math.degrees(math.atan2(cross, dot)))

        if dist_from_last >= max_dist or da >= max_angle_deg:
            out.append(C)

    if not np.allclose(out[-1], out[0], atol=1e-3):
        out.append(out[0])

    return np.array(out, dtype=np.float64)


def smooth_centerline(
    cl: np.ndarray, sigma: float = 3.0, iterations: int = 1
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    pts = cl.copy()
    for _ in range(iterations):
        pts = np.stack(
            [
                gaussian_filter1d(pts[:, 0], sigma, mode="wrap"),
                gaussian_filter1d(pts[:, 1], sigma, mode="wrap"),
            ],
            axis=1,
        )
    return pts


def process_track(img_path: str, json_path: str, out_path: str) -> bool:
    basename = os.path.basename(img_path)
    track_name = os.path.splitext(basename)[0]

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"{track_name}: cannot read image")
        return False

    pts = extract_skeleton(img)
    if pts is None:
        raise RuntimeError(f"{track_name}: skeleton extraction failed")

    arrow_centroid, direction = find_red_arrow(img)
    start_px = find_start_pixel(img, pts, arrow_centroid)

    path = walk_loop(pts, start_px)

    if len(path) < len(pts) * 0.80:
        best_path = path
        step = max(1, len(pts) // 12)
        for seed in sorted(pts, key=lambda p: p[0])[::step]:
            trial = walk_loop(pts, seed)
            if len(trial) > len(best_path):
                best_path = trial
        path = best_path

    if start_px in path:
        si = path.index(start_px)
    else:
        si = min(
            range(len(path)),
            key=lambda i: (path[i][0] - start_px[0]) ** 2
            + (path[i][1] - start_px[1]) ** 2,
        )
    path = path[si:] + path[:si]

    path = iron_out_start_bump(path, radius=35)

    centerline = np.array(path, dtype=np.float64)
    if len(centerline) < 10:
        raise RuntimeError(f"{track_name}: too few points")
        return False

    if direction is not None:
        fwd = centerline[min(50, len(centerline) - 1)] - centerline[0]
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm > 0 and np.dot(direction, fwd / fwd_norm) < 0:
            centerline = centerline[::-1]
            d = centerline - np.array(start_px, dtype=np.float64)
            si = np.einsum("ij,ij->i", d, d).argmin()
            centerline = np.roll(centerline, -si, axis=0)

    def scale_to_size(points, size):
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)

        extent = max_xy - min_xy
        scale = size / max(1.0, extent.max())

        return (points - min_xy) * scale

    centerline = scale_to_size(centerline, 8000.0)

    centerline = smooth_centerline(centerline, sigma=3.0)

    centerline = np.vstack([centerline, centerline[0]])

    # centerline = resample_adaptive(centerline,
    #                                min_dist=10,
    #                                max_dist=100.0,
    #                                max_angle_deg=2.5)

    max_spacing = 30.0
    segs = np.hypot(np.diff(centerline[:, 0]), np.diff(centerline[:, 1]))
    total_len = segs.sum()
    num_points = int(total_len / max_spacing)
    centerline = resample_uniform(centerline, num_points=num_points)

    if np.allclose(centerline[0], centerline[-1], atol=1e-3):
        centerline = centerline[:-1]

    with open(json_path, "r") as f:
        labels = json.load(f)

    if labels[f"{basename}"]["reverse"]:
        centerline = centerline[::-1]
    else:
        centerline = np.roll(centerline, -1, axis=0)

    n = len(centerline)

    ROAD_HALF_WIDTH = 40.5
    GRASS_WIDTH = 40
    le, re, lge, rge, cw = generate_edges(centerline, ROAD_HALF_WIDTH, GRASS_WIDTH)

    with open(out_path, "w") as f:
        f.write("DIRECTION CW\n")
        f.write(f"N_VERTICES {n}\n")
        for i in range(n):
            f.write(
                f"{le[i][0]:.2f} {le[i][1]:.2f} "
                f"{re[i][0]:.2f} {re[i][1]:.2f} "
                f"{lge[i][0]:.2f} {lge[i][1]:.2f} "
                f"{rge[i][0]:.2f} {rge[i][1]:.2f} "
                f"{centerline[i][0]:.2f} {centerline[i][1]:.2f}\n"
            )

    canvas = np.full((4000, 4000, 3), 245, dtype=np.uint8)

    def ipt(p):
        return (int(round(p[0])), int(round(p[1])))

    for i in range(n):
        p1 = ipt(centerline[i])
        p2 = ipt(centerline[(i + 1) % n])
        cv2.line(canvas, p1, p2, (40, 40, 40), 6)

    for i in range(n):
        t = i / n
        color = (int(255 * (1 - t)), 60, int(255 * t))
        cv2.circle(canvas, ipt(centerline[i]), 5, color, -1)

    label_step = max(1, n // 20)
    for i in range(0, n, label_step):
        pos = ipt(centerline[i])
        cv2.circle(canvas, pos, 14, (255, 255, 255), -1)
        cv2.circle(canvas, pos, 14, (0, 0, 0), 2)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(
            canvas,
            label,
            (pos[0] - tw // 2, pos[1] + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    arrow_step = max(80, n // 60)
    arrow_len = 80
    for i in range(0, n, arrow_step):
        p1 = centerline[i]
        p2 = centerline[(i + 10) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            continue
        dx /= mag
        dy /= mag
        start_arr = (int(p1[0]), int(p1[1]))
        end_arr = (int(p1[0] + dx * arrow_len), int(p1[1] + dy * arrow_len))
        cv2.arrowedLine(canvas, start_arr, end_arr, (0, 0, 0), 5, tipLength=0.5)

    sp = ipt(centerline[0])
    cv2.circle(canvas, sp, 28, (0, 0, 0), 3)
    cv2.circle(canvas, sp, 22, (0, 200, 0), -1)
    cv2.putText(
        canvas,
        "START",
        (sp[0] - 80, sp[1] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 130, 0),
        3,
        cv2.LINE_AA,
    )

    file_base = os.path.splitext(out_path)[0]
    new_file = file_base + "_trace.png"

    cv2.imwrite(new_file, canvas)
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_img> <output_img>")
        return
    img_path = sys.argv[1]
    json_path = sys.argv[2]
    out_path = sys.argv[3]
    process_track(img_path, json_path, out_path)


if __name__ == "__main__":
    main()
