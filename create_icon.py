#!/usr/bin/env python3
"""Generate a flat spectrum-disc icon for Chromatic Set (12-step color wheel = 12 pitch classes)."""
import os
import math
import subprocess
from PIL import Image, ImageDraw
import colorsys

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_icon(size):
    img = Image.new("RGBA", (size, size), (13, 13, 16, 255))
    d = ImageDraw.Draw(img)
    cx = cy = size / 2
    r_out = size * 0.40
    r_in = size * 0.20
    # 12 wedges, one per pitch class, flat color
    for pc in range(12):
        a0 = -90 + pc * 30
        a1 = a0 + 30
        hue = pc / 12.0
        rr, gg, bb = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
        col = (int(rr * 255), int(gg * 255), int(bb * 255), 255)
        d.pieslice([cx - r_out, cy - r_out, cx + r_out, cy + r_out], a0, a1, fill=col)
    # punch out the center (donut) with background
    d.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=(13, 13, 16, 255))
    return img


def main():
    iconset = os.path.join(APP_DIR, "AppIcon.iconset")
    os.makedirs(iconset, exist_ok=True)
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for s in sizes:
        img = draw_icon(s)
        img.save(os.path.join(iconset, f"icon_{s}x{s}.png"))
        img.save(os.path.join(iconset, f"icon_{s // 2}x{s // 2}@2x.png"))
    # standard iconset names
    mapping = {
        16: "icon_16x16.png", 32: "icon_32x32.png", 128: "icon_128x128.png",
        256: "icon_256x256.png", 512: "icon_512x512.png",
    }
    for s, name in mapping.items():
        draw_icon(s).save(os.path.join(iconset, name))
        draw_icon(s * 2).save(os.path.join(iconset, name.replace(".png", "@2x.png")))
    icns = os.path.join(APP_DIR, "AppIcon.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True)
    print("wrote", icns)


if __name__ == "__main__":
    main()
