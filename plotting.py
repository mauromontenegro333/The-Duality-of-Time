from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size=22, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def line_plot(path, series, xlabel, ylabel, title, width=1100, height=720, xlim=None, ylim=None):
    """Simple grayscale line plot using PIL.

    series: list of dicts with keys x, y, label, dash(optional), marker(optional).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title = _font(28, True)
    f = _font(20, False)
    f_small = _font(18, False)

    left, right, top, bottom = 120, width - 60, 80, height - 110
    xs_all = np.concatenate([np.asarray(s["x"], dtype=float) for s in series])
    ys_all = np.concatenate([np.asarray(s["y"], dtype=float) for s in series])
    if xlim is None:
        xlim = (float(np.nanmin(xs_all)), float(np.nanmax(xs_all)))
    if ylim is None:
        ypad = 0.08 * (float(np.nanmax(ys_all)) - float(np.nanmin(ys_all)) + 1e-12)
        ylim = (float(np.nanmin(ys_all)) - ypad, float(np.nanmax(ys_all)) + ypad)
    xmin, xmax = xlim
    ymin, ymax = ylim

    def sx(x):
        return left + (np.asarray(x) - xmin) / (xmax - xmin) * (right - left)

    def sy(y):
        return bottom - (np.asarray(y) - ymin) / (ymax - ymin) * (bottom - top)

    # Axes and grid.
    draw.rectangle([left, top, right, bottom], outline=(0, 0, 0), width=2)
    for i in range(6):
        x = left + i * (right - left) / 5
        val = xmin + i * (xmax - xmin) / 5
        draw.line([x, top, x, bottom], fill=(225, 225, 225), width=1)
        draw.text((x - 25, bottom + 10), f"{val:.2g}", font=f_small, fill=(0, 0, 0))
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        val = ymin + i * (ymax - ymin) / 5
        draw.line([left, y, right, y], fill=(225, 225, 225), width=1)
        draw.text((20, y - 10), f"{val:.3g}", font=f_small, fill=(0, 0, 0))

    # Lines in grayscale styles.
    styles = [
        {"fill": (0, 0, 0), "width": 4},
        {"fill": (90, 90, 90), "width": 4},
        {"fill": (150, 150, 150), "width": 4},
        {"fill": (40, 40, 40), "width": 2},
    ]
    for idx, s in enumerate(series):
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        pts = list(zip(sx(x), sy(y)))
        st = styles[idx % len(styles)]
        if s.get("dash"):
            for j in range(len(pts) - 1):
                if j % 2 == 0:
                    draw.line([pts[j], pts[j + 1]], fill=st["fill"], width=st["width"])
        else:
            draw.line(pts, fill=st["fill"], width=st["width"], joint="curve")
        # Legend swatch.
        lx, ly = left + 20, top + 20 + idx * 32
        draw.line([lx, ly + 10, lx + 50, ly + 10], fill=st["fill"], width=st["width"])
        draw.text((lx + 60, ly), s["label"], font=f_small, fill=(0, 0, 0))

    draw.text((left, 25), title, font=f_title, fill=(0, 0, 0))
    draw.text(((left + right) // 2 - 50, height - 55), xlabel, font=f, fill=(0, 0, 0))
    # Rotated y label.
    y_label_img = Image.new("RGBA", (300, 40), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label_img)
    yd.text((0, 0), ylabel, font=f, fill=(0, 0, 0))
    y_label_img = y_label_img.rotate(90, expand=True)
    img.paste(y_label_img, (20, int((top + bottom) / 2 - y_label_img.size[1] / 2)), y_label_img)
    img.save(path)
    return str(path)


def corner_plot(path, samples, names, width=1200, height=1200, max_points=3000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    samples = np.asarray(samples)
    if len(samples) > max_points:
        samples = samples[rng.choice(len(samples), size=max_points, replace=False)]
    n = len(names)
    cell = width // n
    img = Image.new("RGB", (cell * n, cell * n), "white")
    draw = ImageDraw.Draw(img)
    f = _font(14)
    for i in range(n):
        for j in range(n):
            x0, y0 = j * cell, i * cell
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(220, 220, 220))
            if i == j:
                vals = samples[:, j]
                lo, hi = np.percentile(vals, [1, 99])
                bins = np.linspace(lo, hi, 30)
                hist, edges = np.histogram(vals, bins=bins)
                if hist.max() > 0:
                    pts = []
                    for k, h in enumerate(hist):
                        x = x0 + 20 + (edges[k] - lo) / (hi - lo + 1e-12) * (cell - 40)
                        y = y0 + cell - 20 - h / hist.max() * (cell - 45)
                        pts.append((x, y))
                    draw.line(pts, fill=(0, 0, 0), width=2)
                draw.text((x0 + 8, y0 + 8), names[j], font=f, fill=(0, 0, 0))
            elif i > j:
                x = samples[:, j]
                y = samples[:, i]
                xlo, xhi = np.percentile(x, [1, 99])
                ylo, yhi = np.percentile(y, [1, 99])
                px = x0 + 20 + (x - xlo) / (xhi - xlo + 1e-12) * (cell - 40)
                py = y0 + cell - 20 - (y - ylo) / (yhi - ylo + 1e-12) * (cell - 40)
                for xx, yy in zip(px, py):
                    if x0 + 20 <= xx <= x0 + cell - 20 and y0 + 20 <= yy <= y0 + cell - 20:
                        draw.point((float(xx), float(yy)), fill=(0, 0, 0))
    img.save(path)
    return str(path)
