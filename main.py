import json
import math
import os
import queue
import csv
import random
import shutil
import subprocess
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import asdict, dataclass
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from PIL import Image, ImageDraw, ImageTk


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SETTINGS_LOG_DIR = RESULTS_DIR / "setting_log"
RESULTS_DIR.mkdir(exist_ok=True)
SETTINGS_LOG_DIR.mkdir(exist_ok=True)


EXPORT_FORMATS = {
    "auto": "自動",
    "json": "JSON（完全・大きめ）",
    "jsonl": "JSONL（大量向け）",
    "csv": "CSV（軽量）",
    "txt": "TXT（人間用）",
}

EXPORT_FORMAT_BY_LABEL = {label: key for key, label in EXPORT_FORMATS.items()}

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "opencv-python が必要です。次を実行してください:\n"
        "pip install opencv-python pillow numpy"
    ) from exc


@dataclass
class ProcessSettings:
    max_image_size: int = 800
    blur_amount: int = 5
    canny_lower: int = 20
    canny_upper: int = 80
    morphology_kernel: int = 3
    min_contour_length: float = 35.0
    min_contour_area: float = 0.0
    approx_epsilon: float = 0.8
    max_total_formulas: int = 450
    samples_per_segment: int = 18
    line_width: int = 2
    fill_close_kernel: int = 5
    fill_boundary_width: int = 4
    invert_lines: bool = False
    overlay: bool = False
    fill_zones: bool = False
    major_only: bool = False
    x_half_range: float = 10.0


def rgb_to_hex(rgb):
    r, g, b = [int(v) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def format_seconds(seconds):
    if seconds is None or math.isinf(seconds):
        return "--:--:--"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def pil_to_tk_image(img, max_size=(410, 310)):
    display = img.copy()
    display.thumbnail(max_size, Image.LANCZOS)
    return ImageTk.PhotoImage(display)


def normalize_odd_kernel(value):
    value = max(0, int(value))
    if value == 0:
        return 0
    return value if value % 2 == 1 else value + 1


def resize_to_max_side(img_rgb, max_side):
    h, w = img_rgb.shape[:2]
    max_side = max(64, int(max_side))
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img_rgb.copy(), 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def pixel_to_math(point, width, height, x_half_range):
    px, py = float(point[0]), float(point[1])
    scale = (2.0 * x_half_range) / max(width - 1, 1)
    y_half_range = scale * (height - 1) / 2.0
    return np.array([-x_half_range + px * scale, y_half_range - py * scale], dtype=float)


def cubic_expression_string(a, b, c, d):
    return f"{a:.6f}*t^3 + {b:.6f}*t^2 + {c:.6f}*t + {d:.6f}"


def bezier_to_power_basis(seg):
    b0, b1, b2, b3 = [np.asarray(p, dtype=float) for p in seg]
    a = -b0 + 3 * b1 - 3 * b2 + b3
    b = 3 * b0 - 6 * b1 + 3 * b2
    c = -3 * b0 + 3 * b1
    d = b0
    return a, b, c, d


def catmull_rom_closed_to_beziers(points):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return []

    if n == 2:
        p0, p1 = pts[0], pts[1]
        return [(p0, p0 + (p1 - p0) / 3.0, p0 + 2.0 * (p1 - p0) / 3.0, p1)]

    segments = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        b0 = p1
        b1 = p1 + (p2 - p0) / 6.0
        b2 = p2 - (p3 - p1) / 6.0
        b3 = p2
        segments.append((b0, b1, b2, b3))
    return segments


def catmull_rom_open_to_beziers(points):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return []

    segments = []
    for i in range(n - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < n else pts[i + 1]
        b0 = p1
        b1 = p1 + (p2 - p0) / 6.0
        b2 = p2 - (p3 - p1) / 6.0
        b3 = p2
        segments.append((b0, b1, b2, b3))
    return segments


def closed_polyline_to_cubic_segments(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return []

    segments = []
    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        segments.append((p0, p0 + (p1 - p0) / 3.0, p0 + 2.0 * (p1 - p0) / 3.0, p1))
    return segments


def open_polyline_to_cubic_segments(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return []

    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    median_length = float(np.median(lengths)) if len(lengths) else 0.0
    max_jump = min(max(35.0, median_length * 3.0), 90.0)

    segments = []
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        if np.linalg.norm(p1 - p0) > max_jump:
            continue
        segments.append((p0, p0 + (p1 - p0) / 3.0, p0 + 2.0 * (p1 - p0) / 3.0, p1))
    return segments


def open_polyline_to_cubic_segments_unfiltered(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return []

    segments = []
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        segments.append((p0, p0 + (p1 - p0) / 3.0, p0 + 2.0 * (p1 - p0) / 3.0, p1))
    return segments


def sample_cubic_bezier(seg, samples=18):
    b0, b1, b2, b3 = [np.asarray(p, dtype=float) for p in seg]
    t = np.linspace(0.0, 1.0, max(2, int(samples)))[:, None]
    return (
        ((1 - t) ** 3) * b0
        + 3 * ((1 - t) ** 2) * t * b1
        + 3 * (1 - t) * (t**2) * b2
        + (t**3) * b3
    )


def contour_points(contour):
    return contour[:, 0, :].astype(float)


def closed_polyline_length(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    shifted = np.roll(pts, -1, axis=0)
    return float(np.sum(np.linalg.norm(shifted - pts, axis=1)))


def resample_closed_polyline(points, target_count):
    pts = np.asarray(points, dtype=float)
    target_count = max(2, int(target_count))
    if len(pts) <= target_count:
        return pts

    closed = np.vstack([pts, pts[0]])
    deltas = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(np.sum(deltas))
    if total <= 0:
        return pts[:target_count]

    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    distances = np.linspace(0.0, total, target_count, endpoint=False)
    out = []
    for distance in distances:
        idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
        idx = min(idx, len(deltas) - 1)
        span = max(deltas[idx], 1e-9)
        local_t = (distance - cumulative[idx]) / span
        out.append(closed[idx] * (1.0 - local_t) + closed[idx + 1] * local_t)
    return np.asarray(out, dtype=float)


def open_polyline_length(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample_open_polyline(points, target_count):
    pts = np.asarray(points, dtype=float)
    target_count = max(2, int(target_count))
    if len(pts) == target_count:
        return pts
    if len(pts) < 2:
        return pts

    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(deltas))
    if total <= 0:
        return pts[:target_count]

    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    distances = np.linspace(0.0, total, target_count)
    out = []
    for distance in distances:
        idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
        idx = min(idx, len(deltas) - 1)
        span = max(deltas[idx], 1e-9)
        local_t = (distance - cumulative[idx]) / span
        out.append(pts[idx] * (1.0 - local_t) + pts[idx + 1] * local_t)
    return np.asarray(out, dtype=float)


def allocate_formula_budget(contour_entries, budget, min_per_contour=1):
    budget = max(0, int(budget))
    if budget <= 0 or not contour_entries:
        return []

    max_selected = max(1, min(len(contour_entries), budget // max(1, min_per_contour), budget // 8))
    contour_entries = contour_entries[:max_selected]

    selected = []
    remaining = budget
    for entry in contour_entries:
        if remaining < min_per_contour:
            break
        cap = int(entry["max_segments"])
        if cap < min_per_contour:
            continue
        selected.append({"entry": entry, "allocation": min_per_contour, "cap": cap})
        remaining -= min_per_contour

    if not selected:
        return []

    while remaining > 0:
        candidates = [item for item in selected if item["allocation"] < item["cap"]]
        if not candidates:
            break
        weights = np.array([max(item["entry"]["score"], 1.0) for item in candidates], dtype=float)
        desired = weights / float(np.sum(weights)) * remaining
        increments = np.floor(desired).astype(int)
        if int(np.sum(increments)) == 0:
            order = np.argsort(-(desired - increments))
            for idx in order:
                item = candidates[int(idx)]
                if item["allocation"] < item["cap"]:
                    item["allocation"] += 1
                    remaining -= 1
                    break
            continue

        for item, inc in zip(candidates, increments):
            if remaining <= 0:
                break
            room = item["cap"] - item["allocation"]
            add = min(int(inc), room, remaining)
            item["allocation"] += add
            remaining -= add

    return [(item["entry"], int(item["allocation"])) for item in selected]


def build_resampled_segments_from_contours(contour_entries, settings):
    budget = max(1, int(settings.max_total_formulas))
    if not contour_entries:
        return []

    selected_count = min(len(contour_entries), budget)
    selected_entries = contour_entries[:selected_count]
    allocations = np.ones(selected_count, dtype=int)
    remaining = budget - selected_count
    if remaining > 0:
        weights = np.array([max(float(entry["score"]), 1.0) for entry in selected_entries], dtype=float)
        fractional = weights / float(np.sum(weights)) * remaining
        extra = np.floor(fractional).astype(int)
        allocations += extra
        leftover = remaining - int(np.sum(extra))
        order = np.argsort(-(fractional - extra))
        for i in range(leftover):
            allocations[order[i % len(order)]] += 1

    selected_segments = []
    used = 0

    for source_index, (entry, allocation) in enumerate(zip(selected_entries, allocations), start=1):
        remaining = budget - used
        if remaining <= 0:
            break
        allocation = min(int(allocation), remaining)
        if allocation <= 0:
            continue
        pts = entry["contour"][:, 0, :].astype(float)
        if len(pts) < 2:
            continue
        sampled = resample_open_polyline(pts, allocation + 1)
        beziers = open_polyline_to_cubic_segments_unfiltered(sampled)
        if not beziers:
            continue
        if len(beziers) > allocation:
            beziers = beziers[:allocation]
        selected_segments.append(
            {
                "source_index": source_index,
                "length": entry["length"],
                "area": entry["area"],
                "beziers": beziers,
            }
        )
        used += len(beziers)

    return selected_segments


def make_preview_from_array(arr):
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def report_progress(callback, start_time, progress, status, log=None):
    elapsed = time.time() - start_time
    eta = elapsed * (100.0 - progress) / progress if progress > 0 else None
    callback({"type": "progress", "value": progress, "eta": eta, "status": status})
    if log:
        callback({"type": "log", "text": log})


def preprocess_edges(img_rgb, settings):
    # Oil-painting texture creates many false edges. Bilateral smoothing keeps
    # large boundaries while reducing brush-stroke speckles.
    denoised = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=60, sigmaSpace=60)
    gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    blur_kernel = normalize_odd_kernel(settings.blur_amount)
    if blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    edges = cv2.Canny(gray, int(settings.canny_lower), int(settings.canny_upper), L2gradient=True)

    kernel_size = max(0, int(settings.morphology_kernel))
    if kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    edges = remove_small_edge_components(edges, settings)

    return gray, edges


def remove_small_edge_components(edges, settings):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    cleaned = np.zeros_like(edges)

    min_pixels = max(6, int(settings.min_contour_length * 0.20))
    min_span = max(6, int(settings.min_contour_length * 0.18))

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_pixels:
            continue
        if max(w, h) < min_span:
            continue
        cleaned[labels == label] = 255

    return cleaned


def iter_pixel_neighbors(pixel, pixels):
    y, x = pixel
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            candidate = (y + dy, x + dx)
            if candidate in pixels:
                yield candidate


def edge_key(a, b):
    return (a, b) if a <= b else (b, a)


def trace_component_paths(component_mask):
    ys, xs = np.where(component_mask > 0)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    if len(pixels) < 2:
        return []

    neighbors = {pixel: list(iter_pixel_neighbors(pixel, pixels)) for pixel in pixels}
    degrees = {pixel: len(items) for pixel, items in neighbors.items()}
    starts = [pixel for pixel, degree in degrees.items() if degree != 2]
    if not starts:
        starts = [next(iter(pixels))]

    visited_edges = set()
    paths = []

    def walk(start, nxt):
        path = [start]
        prev = start
        curr = nxt
        visited_edges.add(edge_key(prev, curr))

        while True:
            path.append(curr)
            choices = [
                item
                for item in neighbors[curr]
                if item != prev and edge_key(curr, item) not in visited_edges
            ]
            if degrees[curr] != 2 or not choices:
                break
            prev, curr = curr, choices[0]
            visited_edges.add(edge_key(prev, curr))
            if curr == start:
                path.append(curr)
                break

        return path

    for start in starts:
        for nxt in neighbors[start]:
            if edge_key(start, nxt) in visited_edges:
                continue
            path = walk(start, nxt)
            if len(path) >= 2:
                paths.append(path)

    # Closed loops have no endpoints. Trace any remaining unvisited edge.
    for start, items in neighbors.items():
        for nxt in items:
            if edge_key(start, nxt) in visited_edges:
                continue
            path = walk(start, nxt)
            if len(path) >= 2:
                paths.append(path)

    return paths


def extract_curve_entries(edges, settings):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    entries = []

    for label in range(1, num_labels):
        x, y, w, h, component_area = stats[label]
        if settings.major_only and max(w, h) < max(edges.shape[:2]) * 0.08:
            continue

        component_mask = np.where(labels[y : y + h, x : x + w] == label, 255, 0).astype(np.uint8)
        paths = trace_component_paths(component_mask)

        for path in paths:
            pts = np.asarray([[px + x, py + y] for py, px in path], dtype=np.float32)
            length = open_polyline_length(pts)
            if length < settings.min_contour_length:
                continue

            px_min = float(np.min(pts[:, 0]))
            py_min = float(np.min(pts[:, 1]))
            px_max = float(np.max(pts[:, 0]))
            py_max = float(np.max(pts[:, 1]))
            path_w = px_max - px_min + 1.0
            path_h = py_max - py_min + 1.0
            bbox_span = max(path_w, path_h)
            bbox_diag = math.hypot(path_w, path_h)
            area = path_w * path_h
            if settings.min_contour_area > 0 and area < settings.min_contour_area:
                continue
            if bbox_span < settings.min_contour_length * 0.25:
                continue

            epsilon = max(0.0, length * float(settings.approx_epsilon) / 100.0)
            approx = cv2.approxPolyDP(pts.reshape((-1, 1, 2)), epsilon, False)
            approx_pts = contour_points(approx)
            if len(approx_pts) < 2:
                continue

            max_segments = min(max(1, len(approx_pts) - 1), max(2, int(settings.max_total_formulas)))
            score = length + bbox_diag * 3.0 + math.sqrt(max(component_area, 0.0)) * 2.0
            entries.append(
                {
                    "approx_points": approx_pts,
                    "length": length,
                    "area": area,
                    "bbox": [int(px_min), int(py_min), int(path_w), int(path_h)],
                    "max_segments": max_segments,
                    "score": score,
                }
            )

    entries.sort(key=lambda item: item["score"], reverse=True)
    return entries


def extract_contour_entries(edges, settings):
    mode = cv2.RETR_EXTERNAL if settings.major_only else cv2.RETR_LIST
    contours_info = cv2.findContours(edges, mode, cv2.CHAIN_APPROX_NONE)
    contours = contours_info[-2]
    entries = []

    for contour in contours:
        length = float(cv2.arcLength(contour, True))
        area = float(abs(cv2.contourArea(contour)))
        x, y, w, h = cv2.boundingRect(contour)
        bbox_span = float(max(w, h))
        bbox_diag = math.hypot(w, h)
        if length < settings.min_contour_length:
            continue
        if settings.min_contour_area > 0 and area < settings.min_contour_area:
            continue
        if bbox_span < settings.min_contour_length * 0.25:
            continue

        epsilon = max(0.0, length * float(settings.approx_epsilon) / 100.0)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = contour_points(approx)
        if len(pts) < 2:
            continue

        max_segments = min(len(pts), max(2, int(settings.max_total_formulas)))
        score = length + bbox_diag * 2.5 + math.sqrt(max(area, 0.0)) * 4.0
        entries.append(
            {
                "contour": contour,
                "approx_points": pts,
                "length": length,
                "area": area,
                "bbox": [int(x), int(y), int(w), int(h)],
                "max_segments": max_segments,
                "score": score,
            }
        )

    entries.sort(key=lambda item: item["score"], reverse=True)
    return entries


def render_segments(width, height, img_rgb, selected_segments, settings):
    if settings.overlay:
        base = Image.fromarray(img_rgb).convert("RGB")
        line_color = (255, 255, 255) if settings.invert_lines else (0, 0, 0)
    else:
        bg = (0, 0, 0) if settings.invert_lines else (255, 255, 255)
        line_color = (255, 255, 255) if settings.invert_lines else (0, 0, 0)
        base = Image.new("RGB", (width, height), bg)

    draw = ImageDraw.Draw(base)
    for seg in selected_segments:
        pts = []
        for bezier in seg["beziers"]:
            sampled = sample_cubic_bezier(bezier, settings.samples_per_segment)
            if pts:
                sampled = sampled[1:]
            pts.extend((float(x), float(y)) for x, y in sampled)
        if len(pts) >= 2:
            draw.line(pts, fill=line_color, width=max(1, int(settings.line_width)), joint="curve")
    return base


def render_edge_line_image(edges, img_rgb, settings):
    line_mask = thicken_boundary_mask(edges, settings.line_width)

    if settings.overlay:
        base = Image.fromarray(img_rgb).convert("RGB")
        overlay = Image.new("RGB", base.size, (255, 255, 255) if settings.invert_lines else (0, 0, 0))
        mask = Image.fromarray(line_mask).convert("L")
        base.paste(overlay, mask=mask)
        return base

    if settings.invert_lines:
        arr = np.where(line_mask > 0, 255, 0).astype(np.uint8)
    else:
        arr = np.where(line_mask > 0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def thicken_boundary_mask(edges, line_width):
    line_mask = edges.copy()
    width = max(1, int(line_width))
    if width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        line_mask = cv2.dilate(line_mask, kernel, iterations=1)
    return line_mask


def prepare_fill_boundary_mask(edges, settings):
    boundary = edges.copy()

    close_kernel = normalize_odd_kernel(settings.fill_close_kernel)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, kernel)

    boundary = thicken_boundary_mask(boundary, settings.fill_boundary_width)

    # Prevent the outside of the image from leaking into interior regions.
    boundary[0, :] = 255
    boundary[-1, :] = 255
    boundary[:, 0] = 255
    boundary[:, -1] = 255
    return boundary


def render_formula_boundary_mask(width, height, selected_segments, settings):
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw_width = max(1, int(settings.line_width))

    for seg in selected_segments:
        for bezier in seg["beziers"]:
            sampled = sample_cubic_bezier(bezier, settings.samples_per_segment)
            pts = [(float(x), float(y)) for x, y in sampled]
            if len(pts) >= 2:
                draw.line(pts, fill=255, width=draw_width, joint="curve")

    mask = np.array(mask_img, dtype=np.uint8)
    if draw_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (draw_width, draw_width))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def color_zones_from_boundary(boundary_mask, img_rgb, settings):
    free_mask = np.where(boundary_mask > 0, 0, 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_mask, connectivity=4)

    filled = np.zeros_like(img_rgb)
    zone_data = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area <= 0:
            continue
        zone_pixels = labels == label
        color = np.mean(img_rgb[zone_pixels], axis=0)
        color = np.clip(np.round(color), 0, 255).astype(np.uint8)
        filled[zone_pixels] = color
        zone_data.append(
            {
                "zone_index": len(zone_data) + 1,
                "pixel_count": int(area),
                "bbox": [int(x), int(y), int(w), int(h)],
                "fill_rgb": [int(color[0]), int(color[1]), int(color[2])],
                "fill_hex": rgb_to_hex(color),
            }
        )

    line_color = 255 if settings.invert_lines else 0
    filled[boundary_mask > 0] = line_color
    return Image.fromarray(filled).convert("RGB"), zone_data


def overlay_boundary_on_image(base_image, boundary_mask, settings):
    base = base_image.copy().convert("RGB")
    line_color = (255, 255, 255) if settings.invert_lines else (0, 0, 0)
    overlay = Image.new("RGB", base.size, line_color)
    mask = Image.fromarray(boundary_mask).convert("L")
    base.paste(overlay, mask=mask)
    return base


def build_segments_from_contours(contour_entries, settings):
    budget = max(1, int(settings.max_total_formulas))
    selected_segments = []
    used = 0

    # Increase simplification only if the first pass cannot include enough
    # coherent contours. This preserves recognizable local shapes better than
    # forcing every contour into an arbitrary segment allocation.
    for multiplier in (0.12, 0.2, 0.35, 0.55, 0.8, 1.0, 1.4, 2.0, 3.0, 4.5, 6.0):
        candidates = []
        for entry in contour_entries:
            epsilon = max(0.0, entry["length"] * float(settings.approx_epsilon) / 100.0 * multiplier)
            approx = cv2.approxPolyDP(entry["contour"], epsilon, False)
            pts = contour_points(approx)
            if len(pts) < 2:
                continue
            segment_count = len(pts)
            candidates.append((entry, pts, segment_count))

        candidates.sort(key=lambda item: item[0]["score"], reverse=True)
        trial = []
        trial_used = 0
        for entry, pts, segment_count in candidates:
            remaining = budget - trial_used
            if remaining <= 0:
                break
            if segment_count > remaining:
                if remaining < 2:
                    break
                pts = resample_open_polyline(pts, remaining + 1)
                segment_count = len(pts) - 1
            beziers = catmull_rom_open_to_beziers(pts)
            if not beziers:
                continue
            trial.append(
                {
                    "source_index": len(trial) + 1,
                    "length": entry["length"],
                    "area": entry["area"],
                    "beziers": beziers,
                }
            )
            trial_used += len(beziers)
            if trial_used >= budget:
                break

        selected_segments = trial
        used = trial_used
        if used >= budget * 0.75 or multiplier == 6.0:
            break

    return selected_segments


def build_formula_exports(selected_segments, width, height, settings, image_path, scale, zone_data=None):
    formula_data = []
    total = 0

    for contour_index, contour_entry in enumerate(selected_segments, start=1):
        segments = []
        for segment_index, bezier_px in enumerate(contour_entry["beziers"], start=1):
            math_seg = tuple(
                pixel_to_math(point, width, height, settings.x_half_range) for point in bezier_px
            )
            a, b, c, d = bezier_to_power_basis(math_seg)
            segments.append(
                {
                    "segment_index": segment_index,
                    "x_coefficients": [float(a[0]), float(b[0]), float(c[0]), float(d[0])],
                    "y_coefficients": [float(a[1]), float(b[1]), float(c[1]), float(d[1])],
                    "x": cubic_expression_string(a[0], b[0], c[0], d[0]),
                    "y": cubic_expression_string(a[1], b[1], c[1], d[1]),
                    "domain": "0 <= t <= 1",
                    "sampled_points_px": [
                        [float(x), float(y)]
                        for x, y in sample_cubic_bezier(bezier_px, settings.samples_per_segment)
                    ],
                }
            )

        total += len(segments)
        formula_data.append(
            {
                "contour_index": contour_index,
                "source_length_px": float(contour_entry["length"]),
                "source_area_px": float(contour_entry["area"]),
                "allocated_formula_count": len(segments),
                "segments": segments,
            }
        )

    project_meta = {
        "app": "Contour Graph Art Formula Generator",
        "source_image_path": image_path,
        "processed_width": width,
        "processed_height": height,
        "resize_scale_from_original": float(scale),
        "settings": asdict(settings),
        "total_formula_count": total,
        "zone_count": len(zone_data or []),
        "coordinate_system": {
            "x": "centered horizontally",
            "y": "centered vertically, positive upward",
            "segment_domain": "0 <= t <= 1",
            "x_half_range": settings.x_half_range,
        },
    }
    return project_meta, formula_data


def build_formula_text(project_meta, formula_data):
    lines = [
        "=== Contour Graph Art Formula Export ===",
        "",
        f"Source image: {project_meta['source_image_path']}",
        f"Processed size: {project_meta['processed_width']} x {project_meta['processed_height']}",
        f"Total formulas: {project_meta['total_formula_count']}",
        f"Paint zones: {project_meta.get('zone_count', 0)}",
        "",
        "Formula form:",
        "  x(t) = a*t^3 + b*t^2 + c*t + d",
        "  y(t) = e*t^3 + f*t^2 + g*t + h",
        "  domain: 0 <= t <= 1",
        "",
        "Coordinate system:",
        "  origin is image center",
        "  x is positive to the right",
        "  y is positive upward",
        "",
        "Settings:",
    ]

    for key, value in project_meta["settings"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    for contour in formula_data:
        lines.append("--------------------------------------------------")
        lines.append(
            f"Contour {contour['contour_index']} | "
            f"length_px={contour['source_length_px']:.2f} | "
            f"area_px={contour['source_area_px']:.2f} | "
            f"formulas={contour['allocated_formula_count']}"
        )
        for segment in contour["segments"]:
            lines.append(f"  Segment {segment['segment_index']}")
            lines.append(f"    x(t) = {segment['x']}")
            lines.append(f"    y(t) = {segment['y']}")
            lines.append(f"    domain: {segment['domain']}")
        lines.append("")

    return "\n".join(lines)


def _fmt_num(value, sig_figs=4):
    """有効数字 sig_figs 桁で丸め、末尾ゼロを除いた文字列を返す。"""
    if value == 0:
        return "0"
    mag = math.floor(math.log10(abs(value)))
    decimals = max(0, sig_figs - 1 - mag)
    rounded = round(value, decimals)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def _poly_term_latex(coef, power, is_first, sig_figs=4):
    """一項分の LaTeX 文字列を返す。ゼロ項は None。"""
    threshold = 10 ** (-(sig_figs + 1))
    if abs(coef) < threshold:
        return None

    sign = "+" if coef > 0 else "-"
    abs_c = abs(coef)

    var = {3: "t^{3}", 2: "t^{2}", 1: "t", 0: ""}.get(power, f"t^{{{power}}}")

    # 係数が ±1 かつ変数項がある場合は係数を省略
    if power > 0 and abs(abs_c - 1.0) < threshold:
        coef_str = ""
    else:
        coef_str = _fmt_num(abs_c, sig_figs)

    if var and coef_str:
        term = rf"{coef_str}\,{var}"
    else:
        term = var or coef_str

    if is_first:
        return f"-{term}" if sign == "-" else term
    return rf" {sign} {term}"


def format_poly_latex(coeffs, sig_figs=4):
    """[a, b, c, d] → 'a t^3 + b t^2 + c t + d' の LaTeX 文字列。"""
    terms = [(coeffs[0], 3), (coeffs[1], 2), (coeffs[2], 1), (coeffs[3], 0)]
    parts = []
    for coef, power in terms:
        part = _poly_term_latex(coef, power, not parts, sig_figs)
        if part is not None:
            parts.append(part)
    return "".join(parts) if parts else "0"


def build_formula_latex(project_meta, formula_data, sig_figs=4, txt_include_path=None):
    src = project_meta.get("source_image_path", "")
    w = project_meta.get("processed_width", "?")
    h = project_meta.get("processed_height", "?")
    total = project_meta.get("total_formula_count", "?")

    lines = [
        r"\documentclass{article}",
        r"\usepackage{amsmath}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{verbatim}",
        r"\begin{document}",
        r"",
        r"\begin{center}",
        r"  {\Large\bfseries Contour Graph Art --- Parametric Formulas}\\[6pt]",
        rf"  \texttt{{{src}}}\\[2pt]",
        rf"  {w}\,$\times$\,{h}\,px \quad Total curves: {total}",
        r"\end{center}",
        r"",
        r"\bigskip",
        r"All curves are parametric cubic B\'ezier segments with domain $0 \le t \le 1$.",
        r"Coordinate origin is the image centre; $x$ is rightward, $y$ is upward.",
        r"",
    ]

    for contour in formula_data:
        ci = contour["contour_index"]
        length = contour["source_length_px"]
        n_seg = contour["allocated_formula_count"]
        lines.append(r"\bigskip")
        lines.append(
            rf"\noindent\textbf{{Contour {ci}}} "
            rf"\quad length ${length:.1f}$\,px \quad {n_seg} segment(s)\\"
        )
        lines.append(r"\begin{align*}")
        for seg in contour["segments"]:
            si = seg["segment_index"]
            xpoly = format_poly_latex(seg["x_coefficients"], sig_figs)
            ypoly = format_poly_latex(seg["y_coefficients"], sig_figs)
            lines.append(rf"  x_{{{ci},{si}}}(t) &= {xpoly} \\")
            lines.append(rf"  y_{{{ci},{si}}}(t) &= {ypoly} \\[2pt]")
        lines.append(r"\end{align*}")

    if txt_include_path:
        rel = Path(txt_include_path).name
        lines += [
            r"",
            r"\clearpage",
            r"\section*{Appendix: Formula Details (TXT)}",
            rf"\verbatiminput{{{rel}}}",
        ]

    lines.append(r"\end{document}")
    return "\n".join(lines)


def write_formula_latex(path, project_meta, formula_data, sig_figs=4, txt_include_path=None):
    content = build_formula_latex(project_meta, formula_data, sig_figs, txt_include_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def count_formula_segments(formula_data):
    return sum(len(contour.get("segments", [])) for contour in formula_data or [])


def choose_export_format(format_key, formula_count):
    if format_key != "auto":
        return format_key
    if formula_count <= 10000:
        return "json"
    if formula_count <= 200000:
        return "jsonl"
    return "csv"


def format_bytes(size):
    size = max(0, float(size))
    units = ["B", "KB", "MB", "GB", "TB"]
    unit = 0
    while size >= 1024.0 and unit < len(units) - 1:
        size /= 1024.0
        unit += 1
    if unit == 0:
        return f"{int(size)} {units[unit]}"
    return f"{size:.1f} {units[unit]}"


def estimate_export_size(project_meta, formula_data, zone_data, format_key):
    formula_count = count_formula_segments(formula_data)
    zone_count = len(zone_data or [])
    actual_format = choose_export_format(format_key, formula_count)

    # Conservative estimates. Pretty JSON includes sampled points and repeated keys,
    # so it is intentionally much larger than JSONL/CSV.
    per_segment = {
        "json": 900,
        "jsonl": 260,
        "csv": 150,
        "txt": 230,
    }.get(actual_format, 260)
    base = {
        "json": 12000,
        "jsonl": 4000,
        "csv": 600,
        "txt": 3000,
    }.get(actual_format, 1000)
    per_zone = {
        "json": 120,
        "jsonl": 120,
        "csv": 0,
        "txt": 0,
    }.get(actual_format, 0)

    return int(base + formula_count * per_segment + zone_count * per_zone), actual_format


def csv_cell(value):
    return "" if value is None else value


def write_formula_json(path, project_meta, formula_data, zone_data):
    obj = {
        "project_meta": project_meta,
        "formula_data": formula_data,
        "zone_data": zone_data or [],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def write_formula_jsonl(path, project_meta, formula_data, zone_data):
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps({"type": "project_meta", "data": project_meta}, ensure_ascii=False) + "\n")
        for zone in zone_data or []:
            file.write(json.dumps({"type": "zone", "data": zone}, ensure_ascii=False) + "\n")
        for contour in formula_data or []:
            common = {
                "contour_index": contour.get("contour_index"),
                "source_length_px": contour.get("source_length_px"),
                "source_area_px": contour.get("source_area_px"),
            }
            for segment in contour.get("segments", []):
                row = dict(common)
                row.update({
                    "segment_index": segment.get("segment_index"),
                    "x_coefficients": segment.get("x_coefficients"),
                    "y_coefficients": segment.get("y_coefficients"),
                    "domain": segment.get("domain"),
                })
                file.write(json.dumps({"type": "segment", "data": row}, ensure_ascii=False) + "\n")


def write_formula_csv(path, formula_data):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "contour_index",
            "segment_index",
            "x_a",
            "x_b",
            "x_c",
            "x_d",
            "y_a",
            "y_b",
            "y_c",
            "y_d",
            "source_length_px",
            "source_area_px",
            "domain",
        ])
        for contour in formula_data or []:
            contour_index = contour.get("contour_index")
            length = contour.get("source_length_px")
            area = contour.get("source_area_px")
            for segment in contour.get("segments", []):
                x_coeffs = segment.get("x_coefficients") or ["", "", "", ""]
                y_coeffs = segment.get("y_coefficients") or ["", "", "", ""]
                writer.writerow([
                    contour_index,
                    segment.get("segment_index"),
                    *x_coeffs,
                    *y_coeffs,
                    length,
                    area,
                    segment.get("domain"),
                ])


def default_extension_for_format(format_key):
    return {
        "json": ".json",
        "jsonl": ".jsonl",
        "csv": ".csv",
        "txt": ".txt",
    }.get(format_key, ".json")


def filetypes_for_format(format_key):
    if format_key == "json":
        return [("JSONファイル", "*.json")]
    if format_key == "jsonl":
        return [("JSON Linesファイル", "*.jsonl")]
    if format_key == "csv":
        return [("CSVファイル", "*.csv")]
    if format_key == "txt":
        return [("テキストファイル", "*.txt")]
    return [
        ("JSONファイル", "*.json"),
        ("JSON Linesファイル", "*.jsonl"),
        ("CSVファイル", "*.csv"),
        ("テキストファイル", "*.txt"),
    ]


def process_image_to_formulas(img_rgb, image_path, settings, callback):
    start_time = time.time()
    report_progress(callback, start_time, 3, "画像を読み込み中...", "画像を読み込み中...")

    processed_rgb, scale = resize_to_max_side(img_rgb, settings.max_image_size)
    h, w = processed_rgb.shape[:2]

    report_progress(callback, start_time, 13, "前処理中...", "画像を前処理中...")
    gray, edges = preprocess_edges(processed_rgb, settings)

    report_progress(callback, start_time, 30, "輪郭抽出中...", "エッジ輪郭を検出中...")
    contour_entries = extract_contour_entries(edges, settings)
    if not contour_entries:
        raise ValueError("輪郭が見つかりません。Cannyしきい値や最小輪郭フィルタを下げてください。")

    callback({"type": "contours", "rows": contour_entries[:300], "total": len(contour_entries)})
    report_progress(
        callback,
        start_time,
        45,
        "式数を配分中...",
        f"使用可能な輪郭を {len(contour_entries)} 件検出しました。",
    )

    report_progress(callback, start_time, 62, "曲線近似中...")
    selected_segments = build_segments_from_contours(contour_entries, settings)
    report_progress(callback, start_time, 80, "曲線近似中...")

    report_progress(callback, start_time, 84, "プレビュー描画中...", "最終輪郭画像を描画中...")
    zone_data = []
    formula_boundary = render_formula_boundary_mask(w, h, selected_segments, settings)
    if settings.fill_zones:
        boundary = prepare_fill_boundary_mask(formula_boundary, settings)
        filled_image, zone_data = color_zones_from_boundary(boundary, processed_rgb, settings)
        result_image = overlay_boundary_on_image(filled_image, formula_boundary, settings)
    else:
        result_image = render_edge_line_image(formula_boundary, processed_rgb, settings)
    edge_image = make_preview_from_array(edges)
    gray_image = make_preview_from_array(gray)
    processed_image = Image.fromarray(processed_rgb).convert("RGB")

    report_progress(callback, start_time, 93, "数式データ作成中...", "TXT/JSON用の数式データを作成中...")
    project_meta, formula_data = build_formula_exports(
        selected_segments,
        w,
        h,
        settings,
        image_path,
        scale,
        zone_data,
    )
    report_progress(callback, start_time, 100, "書き出し準備完了", "書き出し準備完了。")
    return {
        "processed_image": processed_image,
        "gray_image": gray_image,
        "edge_image": edge_image,
        "result_image": result_image,
        "project_meta": project_meta,
        "formula_data": formula_data,
        "zone_data": zone_data,
    }


def evaluate_result_quality(result_image_pil, original_image_pil):
    """エッジカバレッジと精度からスコアを計算。(score, coverage, precision) を返す。"""
    orig_np = np.array(original_image_pil.convert("L"))
    result_np = np.array(result_image_pil.convert("L"))

    orig_edges = cv2.Canny(orig_np, 20, 60)
    # result は白背景・黒線なので反転して「線ピクセル」を取る
    result_lines = (result_np < 128).astype(np.uint8) * 255

    # 比較のため result をリサイズ
    if orig_edges.shape != result_lines.shape:
        result_lines = cv2.resize(result_lines, (orig_edges.shape[1], orig_edges.shape[0]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_edges = cv2.dilate(orig_edges, kernel)
    dilated_lines = cv2.dilate(result_lines, kernel)

    edge_px = orig_edges > 0
    line_px = result_lines > 0
    dilated_edge_px = dilated_edges > 0
    dilated_line_px = dilated_lines > 0

    coverage = float((edge_px & dilated_line_px).sum()) / max(edge_px.sum(), 1)
    precision = float((line_px & dilated_edge_px).sum()) / max(line_px.sum(), 1)

    if coverage + precision < 1e-6:
        return 0.0, 0.0, 0.0
    score = 2 * coverage * precision / (coverage + precision)
    return score, coverage, precision


def reflect_settings(settings, score, coverage, precision, loop_index):
    """スコア・カバレッジ・精度をもとに次ループのパラメータを調整して返す。"""
    from dataclasses import asdict
    s = ProcessSettings(**asdict(settings))

    if coverage < 0.45 and precision >= 0.5:
        # 線が少ない → 検出を緩める
        s.canny_lower = max(5, int(s.canny_lower * 0.8))
        s.canny_upper = max(s.canny_lower + 20, int(s.canny_upper * 0.85))
        s.min_contour_length = max(8.0, s.min_contour_length * 0.8)
        s.max_total_formulas = min(1_000_000, int(s.max_total_formulas * 1.4))

    elif precision < 0.45 and coverage >= 0.5:
        # ノイズが多い → 検出を絞る
        s.blur_amount = min(15, s.blur_amount + 2)
        s.canny_lower = min(80, int(s.canny_lower * 1.2))
        s.canny_upper = max(s.canny_lower + 20, int(s.canny_upper * 1.15))
        s.min_contour_length = min(120.0, s.min_contour_length * 1.25)

    elif coverage < 0.45 and precision < 0.45:
        # 両方悪い → ぼかしを減らしつつ式数を増やす
        s.blur_amount = max(1, s.blur_amount - 1)
        s.canny_lower = max(5, int(s.canny_lower * 0.85))
        s.min_contour_length = max(8.0, s.min_contour_length * 0.85)
        s.max_total_formulas = min(1_000_000, int(s.max_total_formulas * 1.5))

    else:
        # スコアが良い → さらに精密化
        s.max_total_formulas = min(1_000_000, int(s.max_total_formulas * 1.25))
        s.approx_epsilon = max(0.1, s.approx_epsilon * 0.8)

    return s


# 各パラメータの「絶対NG」限界値。有効な美しい範囲ではなく壊れない限界。
RANDOM_PARAM_LIMITS = {
    "blur_amount":         (0,    31),
    "canny_lower":         (1,   250),
    "canny_upper_delta":   (5,   200),   # canny_upper = canny_lower + delta
    "morphology_kernel":   (0,    21),
    "min_contour_length":  (2.0, 300.0),
    "min_contour_area":    (0.0, 5000.0),
    "approx_epsilon":      (0.05, 15.0),
    "max_total_formulas":  (10, 1_000_000),
    "samples_per_segment": (2,    60),
    "line_width":          (1,    15),
    "fill_close_kernel":   (0,    25),
    "fill_boundary_width": (1,    25),
}


def random_settings(base_settings):
    """絶対NG範囲を外した完全ランダムパラメータを生成する。"""
    L = RANDOM_PARAM_LIMITS
    s = ProcessSettings()
    s.max_image_size = base_settings.max_image_size   # 解像度は元設定を引き継ぐ
    s.x_half_range = base_settings.x_half_range       # 座標系も固定
    s.blur_amount = random.randint(*L["blur_amount"])
    s.canny_lower = random.randint(*L["canny_lower"])
    s.canny_upper = s.canny_lower + random.randint(*L["canny_upper_delta"])
    s.morphology_kernel = random.randint(*L["morphology_kernel"])
    s.min_contour_length = random.uniform(*L["min_contour_length"])
    s.min_contour_area = random.uniform(*L["min_contour_area"])
    s.approx_epsilon = random.uniform(*L["approx_epsilon"])
    s.max_total_formulas = random.randint(*L["max_total_formulas"])
    s.samples_per_segment = random.randint(*L["samples_per_segment"])
    s.line_width = random.randint(*L["line_width"])
    s.fill_close_kernel = random.randint(*L["fill_close_kernel"])
    s.fill_boundary_width = random.randint(*L["fill_boundary_width"])
    s.invert_lines = random.choice([True, False])
    s.overlay = False
    s.fill_zones = random.choice([True, False])
    s.major_only = random.choice([True, False])
    return s


def _parse_settings_file(path):
    """setting_log JSON または 説明 TXT から設定辞書を返す。"""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # setting_log の {"settings": {...}} 形式
        if "settings" in data:
            return data["settings"]
        # project_meta を直接保存した場合も同じキー
        raise ValueError("JSON に 'settings' キーが見つかりません。")
    else:
        # 説明 TXT: "Settings:" 以降の "  key: value" 行を読む
        result = {}
        in_settings = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip()
                if stripped.strip() == "Settings:":
                    in_settings = True
                    continue
                if in_settings:
                    if stripped == "" or (stripped and stripped[0] not in (" ", "\t")):
                        break
                    if ":" in stripped:
                        key, _, val = stripped.strip().partition(": ")
                        result[key.strip()] = val.strip()
        if not result:
            raise ValueError("TXT に Settings セクションが見つかりません。")
        # 型変換: True/False 文字列・数値を適切に変換
        for k, v in result.items():
            if v.lower() == "true":
                result[k] = True
            elif v.lower() == "false":
                result[k] = False
            else:
                try:
                    result[k] = int(v)
                except ValueError:
                    try:
                        result[k] = float(v)
                    except ValueError:
                        pass
        return result


# ── Video processing ─────────────────────────────────────────────────────────

def _video_frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total, fps, w, h


def _open_ffmpeg_writer(output_path, fps, w, h):
    """
    ffmpeg サブプロセスを開いて (proc, write_fn, close_fn) を返す。
    write_fn(bgr_ndarray) でフレームを書き込み、close_fn() で終了。
    ffmpeg が使えない場合は OpenCV VideoWriter にフォールバック。
    """
    output_path = Path(output_path)

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-movflags", "+faststart",
            str(output_path),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        def write_fn(bgr):
            proc.stdin.write(bgr.tobytes())

        def close_fn():
            proc.stdin.close()
            proc.wait()

        return proc, write_fn, close_fn

    # ffmpeg なし → OpenCV fallback（avc1 → mp4v → XVID の順で試す）
    for codec, ext in [("avc1", ".mp4"), ("mp4v", ".mp4"), ("XVID", ".avi")]:
        out = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, writer.write, writer.release
        writer.release()

    raise RuntimeError("動画ライターを開けませんでした（ffmpeg も OpenCV も失敗）")


def process_video_to_video(video_path, output_path, settings,
                           stride, output_fps, callback):
    """
    動画 → フレーム毎に関数近似 → 動画に戻す。
    stride    : 何フレームおきに処理するか（1=全, 2=隔, …）
    output_fps: 0 のとき元動画と同じ fps
    callback  : queue.put 相当
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けません: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    out_fps      = output_fps if output_fps > 0 else src_fps

    writer_handle = None
    write_fn      = None
    close_fn      = None
    start_time    = time.time()
    processed_count = 0
    frame_num       = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % stride == 0:
                pct = min(99, int(frame_num / max(total_frames, 1) * 100))
                elapsed = time.time() - start_time
                eta = elapsed * (100 - pct) / pct if pct > 0 else None
                callback({
                    "type": "video_progress",
                    "frame": frame_num,
                    "total": total_frames,
                    "value": pct,
                    "eta": eta,
                    "status": f"動画フレーム {frame_num + 1}/{total_frames} 処理中...",
                })

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    outputs = process_image_to_formulas(
                        frame_rgb, str(video_path), settings, lambda _: None
                    )
                    result_pil = outputs["result_image"]
                    out_w, out_h = result_pil.size

                    # 最初のフレームでライター初期化
                    if write_fn is None:
                        writer_handle, write_fn, close_fn = _open_ffmpeg_writer(
                            output_path, out_fps, out_w, out_h
                        )

                    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                    write_fn(result_bgr)
                    processed_count += 1

                except Exception as exc:
                    callback({"type": "log", "text": f"フレーム {frame_num} エラー: {exc}"})

            frame_num += 1

    finally:
        cap.release()
        if close_fn is not None:
            close_fn()

    return processed_count, src_fps


class ContourGraphArtApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LineGraphify - 輪郭線グラフアート生成")
        self.geometry("1480x980")
        self.minsize(1220, 820)

        self.original_image_pil = None
        self.original_image_np = None
        self.original_image_path = None
        self.outputs = None
        self.processing_thread = None
        self.queue = queue.Queue()

        self.auto_loop_results = []
        self.gallery_window = None

        self.video_src_path = None   # 入力動画パス
        self.video_thread   = None   # 動画処理スレッド

        self._build_ui()
        self.after(100, self._process_queue)

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="画像読み込み", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="設定読み込み", command=self.load_settings).pack(side=tk.LEFT, padx=4)
        self.run_button = ttk.Button(top, text="処理開始", command=self.start_processing)
        self.run_button.pack(side=tk.LEFT, padx=4)
        self.save_png_button = ttk.Button(top, text="PNG保存", command=self.save_png, state=tk.DISABLED)
        self.save_png_button.pack(side=tk.LEFT, padx=4)
        self.save_txt_button = ttk.Button(top, text="説明TXT保存", command=self.save_txt, state=tk.DISABLED)
        self.save_txt_button.pack(side=tk.LEFT, padx=4)
        self.save_json_button = ttk.Button(top, text="数式データ保存", command=self.save_formula_data, state=tk.DISABLED)
        self.save_json_button.pack(side=tk.LEFT, padx=4)
        self.save_latex_button = ttk.Button(top, text="LaTeX書き出し", command=self.save_latex, state=tk.DISABLED)
        self.save_latex_button.pack(side=tk.LEFT, padx=4)

        self.file_label = ttk.Label(top, text="画像未読み込み")
        self.file_label.pack(side=tk.LEFT, padx=12)

        self._build_settings()
        self._build_auto_loop()
        self._build_video_section()
        self._build_previews()
        self._build_progress_and_logs()

    def _build_settings(self):
        settings = ttk.LabelFrame(self, text="処理パラメータ", padding=8)
        settings.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        self.max_image_size_var = tk.IntVar(value=800)
        self.blur_amount_var = tk.IntVar(value=5)
        self.canny_lower_var = tk.IntVar(value=20)
        self.canny_upper_var = tk.IntVar(value=80)
        self.morphology_kernel_var = tk.IntVar(value=3)
        self.min_contour_length_var = tk.DoubleVar(value=35.0)
        self.min_contour_area_var = tk.DoubleVar(value=0.0)
        self.approx_epsilon_var = tk.DoubleVar(value=0.8)
        self.max_total_formulas_var = tk.IntVar(value=450)
        self.samples_per_segment_var = tk.IntVar(value=18)
        self.line_width_var = tk.IntVar(value=2)
        self.fill_close_kernel_var = tk.IntVar(value=5)
        self.fill_boundary_width_var = tk.IntVar(value=4)
        self.x_half_range_var = tk.DoubleVar(value=10.0)
        self.invert_lines_var = tk.BooleanVar(value=False)
        self.overlay_var = tk.BooleanVar(value=False)
        self.fill_zones_var = tk.BooleanVar(value=False)
        self.major_only_var = tk.BooleanVar(value=False)
        self.export_format_var = tk.StringVar(value=EXPORT_FORMATS["auto"])
        self.size_estimate_var = tk.StringVar(value="予想サイズ: 未生成")

        fields = [
            ("画像最大辺", self.max_image_size_var),
            ("ぼかし量", self.blur_amount_var),
            ("Canny下限", self.canny_lower_var),
            ("Canny上限", self.canny_upper_var),
            ("形態処理カーネル", self.morphology_kernel_var),
            ("最小輪郭長", self.min_contour_length_var),
            ("最小輪郭面積", self.min_contour_area_var),
            ("近似強度 %", self.approx_epsilon_var),
            ("最大式数", self.max_total_formulas_var),
            ("区間サンプル数", self.samples_per_segment_var),
            ("線幅", self.line_width_var),
            ("塗り隙間閉じ", self.fill_close_kernel_var),
            ("塗り境界幅", self.fill_boundary_width_var),
            ("数学x半幅", self.x_half_range_var),
        ]

        for index, (label, variable) in enumerate(fields):
            row = index // 4
            col = (index % 4) * 2
            ttk.Label(settings, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=4)
            ttk.Entry(settings, textvariable=variable, width=10).grid(
                row=row, column=col + 1, sticky="w", padx=4, pady=4
            )

        ttk.Checkbutton(settings, text="白黒反転", variable=self.invert_lines_var).grid(
            row=4, column=0, sticky="w", padx=4, pady=4
        )
        ttk.Checkbutton(settings, text="元画像に重ねる", variable=self.overlay_var).grid(
            row=4, column=2, sticky="w", padx=4, pady=4
        )
        ttk.Checkbutton(settings, text="主要輪郭のみ", variable=self.major_only_var).grid(
            row=4, column=4, sticky="w", padx=4, pady=4
        )
        ttk.Checkbutton(settings, text="領域を塗る", variable=self.fill_zones_var).grid(
            row=4, column=6, sticky="w", padx=4, pady=4
        )

        export_frame = ttk.LabelFrame(self, text="保存設定", padding=8)
        export_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(export_frame, text="数式データ形式").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.export_format_combo = ttk.Combobox(
            export_frame,
            textvariable=self.export_format_var,
            values=list(EXPORT_FORMATS.values()),
            state="readonly",
            width=22,
        )
        self.export_format_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self.export_format_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_size_estimate())
        ttk.Label(export_frame, textvariable=self.size_estimate_var).grid(
            row=0, column=2, sticky="w", padx=12, pady=4
        )

    def _build_previews(self):
        previews = ttk.Frame(self, padding=(8, 0, 8, 8))
        previews.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        self.preview_labels = {}
        for title, key in [
            ("元画像", "original"),
            ("処理対象", "processed"),
            ("抽出エッジ", "edges"),
            ("関数近似結果", "result"),
        ]:
            frame = ttk.LabelFrame(previews, text=title, padding=8)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            label = ttk.Label(frame, text="画像なし", anchor="center")
            label.pack(fill=tk.BOTH, expand=True)
            self.preview_labels[key] = label

    def _build_progress_and_logs(self):
        progress_frame = ttk.LabelFrame(self, text="進捗", padding=8)
        progress_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        self.status_var = tk.StringVar(value="待機中")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.eta_var = tk.StringVar(value="残り時間: --:--:--")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor="w")
        ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100).pack(fill=tk.X, pady=6)
        ttk.Label(progress_frame, textvariable=self.eta_var).pack(anchor="w")

        bottom = ttk.Frame(self, padding=(8, 0, 8, 8))
        bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        contours_box = ttk.LabelFrame(bottom, text="輪郭情報", padding=8)
        contours_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        columns = ("rank", "length", "area", "max_segments")
        self.tree = ttk.Treeview(contours_box, columns=columns, show="headings", height=13)
        for column, heading, width in [
            ("rank", "順位", 60),
            ("length", "長さ", 110),
            ("area", "面積", 110),
            ("max_segments", "最大区間", 120),
        ]:
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor="e")
        self.tree.pack(fill=tk.BOTH, expand=True)

        log_box = ttk.LabelFrame(bottom, text="ログ", padding=8)
        log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.log_text = ScrolledText(log_box, wrap=tk.WORD, height=13)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "準備完了。\n")
        self.log_text.configure(state=tk.DISABLED)

    def _build_auto_loop(self):
        frame = ttk.LabelFrame(self, text="ランダム生成", padding=8)
        frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        self.auto_loop_count_var = tk.IntVar(value=12)
        ttk.Label(frame, text="生成枚数").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(frame, textvariable=self.auto_loop_count_var, width=6).grid(row=0, column=1, sticky="w", padx=4)

        self.auto_thread_count_var = tk.IntVar(value=4)
        ttk.Label(frame, text="並列スレッド数").grid(row=0, column=2, sticky="w", padx=(12, 4))
        ttk.Entry(frame, textvariable=self.auto_thread_count_var, width=4).grid(row=0, column=3, sticky="w", padx=4)

        self.auto_loop_button = ttk.Button(frame, text="ランダム生成開始", command=self.start_auto_loop)
        self.auto_loop_button.grid(row=0, column=4, padx=8)

        self.gallery_button = ttk.Button(frame, text="結果ギャラリーを表示", command=self.show_gallery, state=tk.DISABLED)
        self.gallery_button.grid(row=0, column=5, padx=4)

        self.auto_loop_status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.auto_loop_status_var, foreground="blue").grid(
            row=0, column=6, sticky="w", padx=12
        )

    # ── 動画セクション ────────────────────────────────────────────────────────

    def _build_video_section(self):
        frame = ttk.LabelFrame(self, text="動画処理 (MP4 → 関数アート動画)", padding=8)
        frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        # 入力動画ファイル名ラベル
        self.video_name_var = tk.StringVar(value="動画未選択")
        ttk.Button(frame, text="動画読み込み", command=self.load_video).grid(
            row=0, column=0, padx=4, pady=4
        )
        ttk.Label(frame, textvariable=self.video_name_var, width=22, anchor="w").grid(
            row=0, column=1, sticky="w", padx=4
        )

        # フレーム間隔
        self.video_stride_var = tk.IntVar(value=1)
        ttk.Label(frame, text="フレーム間隔").grid(row=0, column=2, sticky="w", padx=(16, 4))
        ttk.Entry(frame, textvariable=self.video_stride_var, width=4).grid(
            row=0, column=3, padx=4
        )
        ttk.Label(frame, text="(1=全, 2=隔, ...)").grid(row=0, column=4, sticky="w")

        # 出力FPS
        self.video_fps_var = tk.DoubleVar(value=0.0)
        ttk.Label(frame, text="出力FPS").grid(row=0, column=5, sticky="w", padx=(16, 4))
        ttk.Entry(frame, textvariable=self.video_fps_var, width=6).grid(
            row=0, column=6, padx=4
        )
        ttk.Label(frame, text="(0=元と同じ)").grid(row=0, column=7, sticky="w")

        # ボタン
        self.video_run_button = ttk.Button(
            frame, text="動画処理開始", command=self.start_video_processing, state=tk.DISABLED
        )
        self.video_run_button.grid(row=0, column=8, padx=(16, 4), pady=4)

        self.video_stop_button = ttk.Button(
            frame, text="中断", command=self.stop_video_processing, state=tk.DISABLED
        )
        self.video_stop_button.grid(row=0, column=9, padx=4)

        # ステータス
        self.video_status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.video_status_var, foreground="blue").grid(
            row=0, column=10, sticky="w", padx=12
        )

    def load_video(self):
        path = filedialog.askopenfilename(
            title="動画ファイルを選択",
            filetypes=[
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if not path:
            return

        try:
            total, fps, w, h = _video_frame_count(path)
        except Exception as exc:
            messagebox.showerror("エラー", f"動画情報の取得に失敗しました。\n\n{exc}")
            return

        self.video_src_path = path
        self.video_name_var.set(os.path.basename(path))
        self.video_run_button.configure(state=tk.NORMAL)
        self.video_status_var.set("")
        self.log(
            f"動画読み込み: {os.path.basename(path)} "
            f"({w}×{h}, {fps:.1f}fps, {total}フレーム)"
        )

    def start_video_processing(self):
        if not self.video_src_path:
            messagebox.showwarning("警告", "先に動画を読み込んでください。")
            return
        if self.video_thread is not None and self.video_thread.is_alive():
            messagebox.showinfo("情報", "すでに処理中です。")
            return

        # 保存先を先に聞く
        default_name = Path(self.video_src_path).stem + "_art.mp4"
        output_path = filedialog.asksaveasfilename(
            title="出力動画の保存先",
            initialdir=str(RESULTS_DIR),
            initialfile=default_name,
            defaultextension=".mp4",
            filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")],
        )
        if not output_path:
            return

        try:
            settings = self._read_settings()
            stride   = max(1, int(self.video_stride_var.get()))
            out_fps  = max(0.0, float(self.video_fps_var.get()))
        except Exception as exc:
            messagebox.showerror("エラー", f"パラメータが不正です。\n\n{exc}")
            return

        self._video_cancel = False
        self.video_run_button.configure(state=tk.DISABLED)
        self.video_stop_button.configure(state=tk.NORMAL)
        self.run_button.configure(state=tk.DISABLED)
        self.auto_loop_button.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("動画処理中...")
        self.video_status_var.set("処理中...")
        self.log(f"動画処理開始 → {os.path.basename(output_path)}  stride={stride}")

        self.video_thread = threading.Thread(
            target=self._worker_video,
            args=(settings, stride, out_fps, output_path),
            daemon=True,
        )
        self.video_thread.start()

    def stop_video_processing(self):
        self._video_cancel = True
        self.video_status_var.set("中断中...")
        self.log("動画処理を中断します…")

    def _worker_video(self, settings, stride, out_fps, output_path):
        def cb(msg):
            # 中断フラグが立ったら poison pill を投げて停止
            if getattr(self, "_video_cancel", False):
                raise InterruptedError("ユーザーが中断しました")
            self.queue.put(msg)

        try:
            count, src_fps = process_video_to_video(
                self.video_src_path, output_path,
                settings, stride, out_fps, cb,
            )
            self.queue.put({
                "type": "video_done",
                "count": count,
                "src_fps": src_fps,
                "output_path": str(output_path),
            })
        except InterruptedError:
            self.queue.put({"type": "video_cancelled"})
        except Exception as exc:
            self.queue.put({"type": "video_error", "text": str(exc)})

    # ── ログ ──────────────────────────────────────────────────────────────────

    def log(self, text):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def load_settings(self):
        path = filedialog.askopenfilename(
            title="設定ファイルを選択（setting_log JSON または 説明 TXT）",
            initialdir=str(RESULTS_DIR),
            filetypes=[
                ("設定ファイル", "*.json *.txt"),
                ("JSON", "*.json"),
                ("テキスト", "*.txt"),
                ("すべて", "*.*"),
            ],
        )
        if not path:
            return
        try:
            settings = _parse_settings_file(path)
            self._apply_settings(settings)
            self.log(f"設定を読み込みました: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("エラー", f"設定の読み込みに失敗しました。\n\n{exc}")

    def _apply_settings(self, d):
        def set_if(var, key, cast):
            if key in d:
                try:
                    var.set(cast(d[key]))
                except (ValueError, TypeError):
                    pass

        set_if(self.max_image_size_var,     "max_image_size",      int)
        set_if(self.blur_amount_var,         "blur_amount",         int)
        set_if(self.canny_lower_var,         "canny_lower",         int)
        set_if(self.canny_upper_var,         "canny_upper",         int)
        set_if(self.morphology_kernel_var,   "morphology_kernel",   int)
        set_if(self.min_contour_length_var,  "min_contour_length",  float)
        set_if(self.min_contour_area_var,    "min_contour_area",    float)
        set_if(self.approx_epsilon_var,      "approx_epsilon",      float)
        set_if(self.max_total_formulas_var,  "max_total_formulas",  int)
        set_if(self.samples_per_segment_var, "samples_per_segment", int)
        set_if(self.line_width_var,          "line_width",          int)
        set_if(self.fill_close_kernel_var,   "fill_close_kernel",   int)
        set_if(self.fill_boundary_width_var, "fill_boundary_width", int)
        set_if(self.x_half_range_var,        "x_half_range",        float)
        set_if(self.invert_lines_var,        "invert_lines",        lambda v: str(v).lower() in ("true", "1"))
        set_if(self.overlay_var,             "overlay",             lambda v: str(v).lower() in ("true", "1"))
        set_if(self.fill_zones_var,          "fill_zones",          lambda v: str(v).lower() in ("true", "1"))
        set_if(self.major_only_var,          "major_only",          lambda v: str(v).lower() in ("true", "1"))

    def load_image(self):
        path = filedialog.askopenfilename(
            title="画像を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if not path:
            return

        try:
            img = Image.open(path).convert("RGB")
            self.original_image_pil = img
            self.original_image_np = np.array(img)
            self.original_image_path = path
            self.outputs = None
            self.file_label.configure(text=os.path.basename(path))
            self._set_preview("original", img)
            for key in ("processed", "edges", "result"):
                self.preview_labels[key].configure(image="", text="画像なし")
                self.preview_labels[key].image = None
            self._set_save_state(tk.DISABLED)
            self.update_size_estimate()
            self._clear_tree()
            self.progress_var.set(0)
            self.status_var.set("画像読み込み完了")
            self.eta_var.set("残り時間: --:--:--")
            self.log(f"画像を読み込みました: {path}")
        except Exception as exc:
            messagebox.showerror("エラー", f"画像の読み込みに失敗しました。\n\n{exc}")

    def _read_settings(self):
        settings = ProcessSettings(
            max_image_size=int(self.max_image_size_var.get()),
            blur_amount=int(self.blur_amount_var.get()),
            canny_lower=int(self.canny_lower_var.get()),
            canny_upper=int(self.canny_upper_var.get()),
            morphology_kernel=int(self.morphology_kernel_var.get()),
            min_contour_length=float(self.min_contour_length_var.get()),
            min_contour_area=float(self.min_contour_area_var.get()),
            approx_epsilon=float(self.approx_epsilon_var.get()),
            max_total_formulas=int(self.max_total_formulas_var.get()),
            samples_per_segment=int(self.samples_per_segment_var.get()),
            line_width=int(self.line_width_var.get()),
            fill_close_kernel=int(self.fill_close_kernel_var.get()),
            fill_boundary_width=int(self.fill_boundary_width_var.get()),
            invert_lines=bool(self.invert_lines_var.get()),
            overlay=bool(self.overlay_var.get()),
            fill_zones=bool(self.fill_zones_var.get()),
            major_only=bool(self.major_only_var.get()),
            x_half_range=float(self.x_half_range_var.get()),
        )
        if settings.max_image_size < 64:
            raise ValueError("画像最大辺は64以上にしてください。")
        if settings.canny_lower < 0 or settings.canny_upper <= settings.canny_lower:
            raise ValueError("Canny上限はCanny下限より大きくしてください。")
        if settings.max_total_formulas <= 0:
            raise ValueError("最大式数は1以上にしてください。")
        if settings.samples_per_segment < 2:
            raise ValueError("区間サンプル数は2以上にしてください。")
        if settings.line_width <= 0:
            raise ValueError("線幅は1以上にしてください。")
        if settings.fill_close_kernel < 0:
            raise ValueError("塗り隙間閉じは0以上にしてください。")
        if settings.fill_boundary_width <= 0:
            raise ValueError("塗り境界幅は1以上にしてください。")
        if settings.x_half_range <= 0:
            raise ValueError("数学x半幅は正の値にしてください。")
        return settings

    def start_processing(self):
        if self.original_image_np is None:
            messagebox.showwarning("警告", "先に画像を読み込んでください。")
            return
        if self.processing_thread is not None and self.processing_thread.is_alive():
            messagebox.showinfo("情報", "すでに処理中です。")
            return

        try:
            settings = self._read_settings()
        except Exception as exc:
            messagebox.showerror("エラー", f"パラメータが不正です。\n\n{exc}")
            return

        self.run_button.configure(state=tk.DISABLED)
        self._set_save_state(tk.DISABLED)
        self._clear_tree()
        self.progress_var.set(0.0)
        self.status_var.set("開始中...")
        self.eta_var.set("残り時間: --:--:--")
        self.log("輪郭線ベースの処理を開始します。")

        self.processing_thread = threading.Thread(
            target=self._worker_process,
            args=(settings,),
            daemon=True,
        )
        self.processing_thread.start()

    def _worker_process(self, settings):
        try:
            outputs = process_image_to_formulas(
                self.original_image_np.copy(),
                self.original_image_path,
                settings,
                self.queue.put,
            )
            self.queue.put({"type": "finished", "outputs": outputs})
        except Exception as exc:
            self.queue.put({"type": "error", "text": str(exc)})

    def start_auto_loop(self):
        if self.original_image_np is None:
            messagebox.showwarning("警告", "先に画像を読み込んでください。")
            return
        if self.processing_thread is not None and self.processing_thread.is_alive():
            messagebox.showinfo("情報", "すでに処理中です。")
            return
        try:
            base_settings = self._read_settings()
            count = max(1, int(self.auto_loop_count_var.get()))
            threads = max(1, min(int(self.auto_thread_count_var.get()), count))
        except Exception as exc:
            messagebox.showerror("エラー", f"パラメータが不正です。\n\n{exc}")
            return

        self.auto_loop_results = []
        self.auto_loop_button.configure(state=tk.DISABLED)
        self.run_button.configure(state=tk.DISABLED)
        self.gallery_button.configure(state=tk.DISABLED)
        self._set_save_state(tk.DISABLED)
        self.auto_loop_status_var.set(f"0/{count} 完了")
        self.log(f"ランダム生成開始: {count} 枚 / {threads} スレッド")

        self.processing_thread = threading.Thread(
            target=self._worker_random_loop,
            args=(base_settings, count, threads),
            daemon=True,
        )
        self.processing_thread.start()

    def _worker_random_loop(self, base_settings, count, thread_count):
        completed_count = 0
        lock = threading.Lock()
        img_np = self.original_image_np.copy()

        def run_one(index):
            nonlocal completed_count
            settings = random_settings(base_settings)
            try:
                outputs = process_image_to_formulas(
                    img_np.copy(),
                    self.original_image_path,
                    settings,
                    lambda _msg: None,  # progress メッセージは捨てる
                )
                score, coverage, precision = evaluate_result_quality(
                    outputs["result_image"],
                    outputs["processed_image"],
                )
                entry = {
                    "iteration": index,
                    "settings": settings,
                    "outputs": outputs,
                    "score": score,
                    "coverage": coverage,
                    "precision": precision,
                }
            except Exception as exc:
                entry = None
                self.queue.put({"type": "log", "text": f"#{index} エラー: {exc}"})

            with lock:
                completed_count += 1
                done = completed_count
                if entry:
                    self.auto_loop_results.append(entry)
                    self.queue.put({
                        "type": "auto_loop_result",
                        "iteration": index,
                        "total": count,
                        "completed": done,
                        "outputs": outputs,
                        "score": score,
                        "coverage": coverage,
                        "precision": precision,
                    })
                else:
                    self.queue.put({
                        "type": "auto_loop_progress",
                        "completed": done,
                        "total": count,
                    })

        with ThreadPoolExecutor(max_workers=thread_count) as pool:
            list(as_completed([pool.submit(run_one, i + 1) for i in range(count)]))

        self.queue.put({"type": "auto_loop_done", "total": count})

    def show_gallery(self):
        if not self.auto_loop_results:
            messagebox.showinfo("情報", "ループ結果がありません。")
            return

        if self.gallery_window and self.gallery_window.winfo_exists():
            self.gallery_window.lift()
            return

        win = tk.Toplevel(self)
        win.title("自動ループ ギャラリー")
        self.gallery_window = win

        canvas = tk.Canvas(win)
        scrollbar = ttk.Scrollbar(win, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        best_score = max(r["score"] for r in self.auto_loop_results)
        thumb_size = (260, 260)

        for result in self.auto_loop_results:
            is_best = abs(result["score"] - best_score) < 1e-6
            label_text = f"#{result['iteration']}  スコア: {result['score']:.3f}"
            if is_best:
                label_text += "  ★ベスト"
            col = ttk.LabelFrame(inner, text=label_text, padding=6)
            col.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)

            img = result["outputs"]["result_image"]
            photo = pil_to_tk_image(img, max_size=thumb_size)
            img_label = ttk.Label(col, image=photo)
            img_label.image = photo
            img_label.pack()

            s = result["settings"]
            info = (
                f"カバレッジ: {result['coverage']:.3f}  精度: {result['precision']:.3f}\n"
                f"式数: {s.max_total_formulas}  Canny: {s.canny_lower}/{s.canny_upper}\n"
                f"ぼかし: {s.blur_amount}  最小輪郭: {s.min_contour_length:.0f}\n"
                f"近似強度: {s.approx_epsilon:.2f}"
            )
            ttk.Label(col, text=info, justify="left", font=("", 9)).pack(anchor="w", pady=(4, 0))

            def make_save_cmd(r=result):
                def cmd():
                    path = filedialog.asksaveasfilename(
                        title="PNG保存",
                        initialdir=str(RESULTS_DIR),
                        defaultextension=".png",
                        filetypes=[("PNGファイル", "*.png")],
                    )
                    if path:
                        r["outputs"]["result_image"].save(path)
                        s = r["settings"]
                        log_path = SETTINGS_LOG_DIR / f"{Path(path).stem}.json"
                        log = {
                            "png_path": str(path),
                            "loop_iteration": r["iteration"],
                            "score": r["score"],
                            "coverage": r["coverage"],
                            "precision": r["precision"],
                            "settings": asdict(s),
                        }
                        with open(log_path, "w", encoding="utf-8") as f:
                            json.dump(log, f, ensure_ascii=False, indent=2)
                return cmd

            ttk.Button(col, text="PNG保存", command=make_save_cmd()).pack(pady=(4, 0))

        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        win.geometry(f"{min(1400, len(self.auto_loop_results) * 290 + 40)}x520")

    def _process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                msg_type = msg.get("type")

                if msg_type == "log":
                    self.log(msg["text"])
                elif msg_type == "progress":
                    self.progress_var.set(msg["value"])
                    self.status_var.set(msg["status"])
                    self.eta_var.set(f"残り時間: {format_seconds(msg['eta'])}")
                elif msg_type == "contours":
                    self._populate_contours(msg["rows"], msg["total"])
                elif msg_type == "finished":
                    self.outputs = msg["outputs"]
                    self._set_preview("processed", self.outputs["processed_image"])
                    self._set_preview("edges", self.outputs["edge_image"])
                    self._set_preview("result", self.outputs["result_image"])
                    self.progress_var.set(100.0)
                    self.status_var.set("書き出し準備完了")
                    self.eta_var.set("残り時間: 00:00:00")
                    self.run_button.configure(state=tk.NORMAL)
                    self._set_save_state(tk.NORMAL)
                    self.update_size_estimate()
                    total = self.outputs["project_meta"]["total_formula_count"]
                    self.log(f"完了しました。総式数: {total}")
                elif msg_type == "video_progress":
                    self.progress_var.set(msg["value"])
                    self.status_var.set(msg["status"])
                    self.eta_var.set(f"残り時間: {format_seconds(msg['eta'])}")
                    done = msg["frame"] + 1
                    tot  = msg["total"]
                    self.video_status_var.set(f"{done}/{tot} フレーム")
                elif msg_type == "video_done":
                    self._video_done_ui()
                    self.log(
                        f"動画処理完了: {msg['count']} フレーム処理 "
                        f"→ {os.path.basename(msg['output_path'])} "
                        f"(元fps: {msg['src_fps']:.1f})"
                    )
                    self.video_status_var.set(
                        f"完了 {msg['count']}フレーム → {os.path.basename(msg['output_path'])}"
                    )
                elif msg_type == "video_cancelled":
                    self._video_done_ui()
                    self.video_status_var.set("中断しました")
                    self.log("動画処理を中断しました。")
                elif msg_type == "video_error":
                    self._video_done_ui()
                    self.video_status_var.set("エラー")
                    self.log("動画処理エラー: " + msg["text"])
                    messagebox.showerror("動画処理エラー", msg["text"])
                elif msg_type == "error":
                    self.run_button.configure(state=tk.NORMAL)
                    self.auto_loop_button.configure(state=tk.NORMAL)
                    self.status_var.set("エラー")
                    self.log("エラー: " + msg["text"])
                    messagebox.showerror("処理エラー", msg["text"])
                elif msg_type == "auto_loop_progress":
                    done, total = msg["completed"], msg["total"]
                    self.auto_loop_status_var.set(f"{done}/{total} 完了")
                    self.progress_var.set(done / total * 100)
                elif msg_type == "auto_loop_result":
                    done, total = msg["completed"], msg["total"]
                    self.auto_loop_status_var.set(f"{done}/{total} 完了")
                    self.progress_var.set(done / total * 100)
                    self._set_preview("result", msg["outputs"]["result_image"])
                    self.log(
                        f"#{msg['iteration']} 完了 — スコア: {msg['score']:.3f} "
                        f"(カバレッジ: {msg['coverage']:.3f} / 精度: {msg['precision']:.3f})"
                    )
                elif msg_type == "auto_loop_done":
                    total = msg["total"]
                    self.run_button.configure(state=tk.NORMAL)
                    self.auto_loop_button.configure(state=tk.NORMAL)
                    self.gallery_button.configure(state=tk.NORMAL)
                    self.progress_var.set(100.0)
                    if self.auto_loop_results:
                        best = max(self.auto_loop_results, key=lambda r: r["score"])
                        self.outputs = best["outputs"]
                        self._set_preview("processed", self.outputs["processed_image"])
                        self._set_preview("edges", self.outputs["edge_image"])
                        self._set_preview("result", self.outputs["result_image"])
                        self._set_save_state(tk.NORMAL)
                        self.update_size_estimate()
                        self.auto_loop_status_var.set(
                            f"完了 {len(self.auto_loop_results)}/{total} — ベスト: #{best['iteration']} スコア {best['score']:.3f}"
                        )
                        self.log(f"ランダム生成完了。{len(self.auto_loop_results)}/{total} 枚成功。ベスト: #{best['iteration']} (スコア {best['score']:.3f})")
        except queue.Empty:
            pass

        self.after(100, self._process_queue)

    def _set_preview(self, key, image):
        photo = pil_to_tk_image(image)
        self.preview_labels[key].configure(image=photo, text="")
        self.preview_labels[key].image = photo

    def _set_save_state(self, state):
        self.save_png_button.configure(state=state)
        self.save_txt_button.configure(state=state)
        self.save_json_button.configure(state=state)
        self.save_latex_button.configure(state=state)

    def _video_done_ui(self):
        self.video_run_button.configure(state=tk.NORMAL)
        self.video_stop_button.configure(state=tk.DISABLED)
        self.run_button.configure(state=tk.NORMAL)
        self.auto_loop_button.configure(state=tk.NORMAL)
        self.progress_var.set(100.0)
        self.status_var.set("待機中")

    def _selected_export_format(self):
        return EXPORT_FORMAT_BY_LABEL.get(self.export_format_var.get(), "auto")

    def update_size_estimate(self):
        if not self.outputs:
            self.size_estimate_var.set("予想サイズ: 未生成")
            return

        format_key = self._selected_export_format()
        estimated, actual_format = estimate_export_size(
            self.outputs["project_meta"],
            self.outputs["formula_data"],
            self.outputs.get("zone_data", []),
            format_key,
        )
        self.size_estimate_var.set(
            f"予想サイズ: 約 {format_bytes(estimated)} / 実形式: {EXPORT_FORMATS[actual_format]}"
        )

    def _clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _populate_contours(self, rows, total):
        self._clear_tree()
        for rank, entry in enumerate(rows, start=1):
            self.tree.insert(
                "",
                tk.END,
                values=(
                    rank,
                    f"{entry['length']:.1f}",
                    f"{entry['area']:.1f}",
                    entry["max_segments"],
                ),
            )
        self.log(f"使用可能な輪郭 {total} 件中、上位 {len(rows)} 件を表示しています。")

    def save_png(self):
        if not self.outputs:
            messagebox.showwarning("警告", "保存する結果がありません。")
            return
        path = filedialog.asksaveasfilename(
            title="PNGを保存",
            initialdir=str(RESULTS_DIR),
            defaultextension=".png",
            filetypes=[("PNGファイル", "*.png")],
        )
        if not path:
            return
        try:
            self.outputs["result_image"].save(path)
            self._save_setting_log(path)
            self.log(f"PNGを保存しました: {path}")
        except Exception as exc:
            messagebox.showerror("エラー", f"PNG保存に失敗しました。\n\n{exc}")

    def _save_setting_log(self, png_path):
        png_stem = Path(png_path).stem
        log_path = SETTINGS_LOG_DIR / f"{png_stem}.json"
        meta = self.outputs.get("project_meta", {})
        settings = meta.get("settings", {})
        log = {
            "png_path": str(png_path),
            "source_image": meta.get("source_image_path"),
            "total_formula_count": meta.get("total_formula_count"),
            "processed_size": [meta.get("processed_width"), meta.get("processed_height")],
            "settings": settings,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        self.log(f"設定ログを保存しました: {log_path}")

    def save_txt(self):
        if not self.outputs:
            messagebox.showwarning("警告", "保存する数式データがありません。")
            return
        path = filedialog.asksaveasfilename(
            title="説明TXTを保存",
            defaultextension=".txt",
            filetypes=[("テキストファイル", "*.txt")],
        )
        if not path:
            return
        try:
            text = build_formula_text(self.outputs["project_meta"], self.outputs["formula_data"])
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
            self.log(f"説明TXTを保存しました: {path}")
        except Exception as exc:
            messagebox.showerror("エラー", f"TXT保存に失敗しました。\n\n{exc}")

    def save_formula_data(self):
        if not self.outputs:
            messagebox.showwarning("警告", "保存する数式データがありません。")
            return
        requested_format = self._selected_export_format()
        actual_format = choose_export_format(
            requested_format,
            self.outputs["project_meta"]["total_formula_count"],
        )
        path = filedialog.asksaveasfilename(
            title="数式データを保存",
            defaultextension=default_extension_for_format(actual_format),
            filetypes=filetypes_for_format(actual_format),
        )
        if not path:
            return
        try:
            if actual_format == "json":
                write_formula_json(
                    path,
                    self.outputs["project_meta"],
                    self.outputs["formula_data"],
                    self.outputs.get("zone_data", []),
                )
            elif actual_format == "jsonl":
                write_formula_jsonl(
                    path,
                    self.outputs["project_meta"],
                    self.outputs["formula_data"],
                    self.outputs.get("zone_data", []),
                )
            elif actual_format == "csv":
                write_formula_csv(path, self.outputs["formula_data"])
            elif actual_format == "txt":
                text = build_formula_text(self.outputs["project_meta"], self.outputs["formula_data"])
                with open(path, "w", encoding="utf-8") as file:
                    file.write(text)
            else:
                raise ValueError(f"未対応の保存形式です: {actual_format}")

            estimated, _ = estimate_export_size(
                self.outputs["project_meta"],
                self.outputs["formula_data"],
                self.outputs.get("zone_data", []),
                actual_format,
            )
            self.log(
                f"数式データを保存しました: {path} "
                f"({EXPORT_FORMATS.get(actual_format, actual_format)}, 予想 {format_bytes(estimated)})"
            )
        except Exception as exc:
            messagebox.showerror("エラー", f"数式データ保存に失敗しました。\n\n{exc}")

    def save_latex(self):
        if not self.outputs:
            messagebox.showwarning("警告", "保存する数式データがありません。")
            return
        path = filedialog.asksaveasfilename(
            title="LaTeX ファイルを保存",
            initialdir=str(RESULTS_DIR),
            defaultextension=".tex",
            filetypes=[("LaTeX ファイル", "*.tex"), ("すべてのファイル", "*.*")],
        )
        if not path:
            return

        txt_path = filedialog.askopenfilename(
            title="インクルードする説明TXTを選択（不要ならキャンセル）",
            initialdir=str(RESULTS_DIR),
            filetypes=[("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")],
        )

        try:
            write_formula_latex(
                path,
                self.outputs["project_meta"],
                self.outputs["formula_data"],
                txt_include_path=txt_path or None,
            )
            total = self.outputs["project_meta"]["total_formula_count"]
            note = f"  + {Path(txt_path).name}" if txt_path else ""
            self.log(f"LaTeX を保存しました: {path}  ({total} curves){note}")
        except Exception as exc:
            messagebox.showerror("エラー", f"LaTeX 保存に失敗しました。\n\n{exc}")


if __name__ == "__main__":
    app = ContourGraphArtApp()
    app.mainloop()
