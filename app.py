#!/usr/bin/env python3
"""
Chromatic Set - turn an image's pixel color field into a ~2-minute composition
built on pitch-class set theory.

Pipeline (see README):
  every pixel -> area-pool to a 2^k Hilbert grid -> gaussian pre-blend ->
  Hilbert traversal -> overlapping windows -> per-window pitch-class SET ->
  Forte analysis + voiced multi-part composition -> JSON + MIDI.

A pixel never becomes its own note. It influences the piece by pooling with its
neighbors (box resize), bleeding into them (blur), and being smeared across the
overlapping windows it falls in. Note rate is a musical dial, decoupled from
pixel count; pixel count controls smoothness.
"""
try:
    import setproctitle
    setproctitle.setproctitle("Chromatic Set")
except ImportError:
    pass  # cosmetic process name only; never block startup on it
import io
import os
import json
import math
import struct
import base64

import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, send_from_directory, send_file

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(APP_DIR, "web")
OUT_DIR = os.path.join(APP_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)
PORT = 5018

app = Flask(__name__, static_folder=None)

# ----------------------------------------------------------------------------
# Note names (pitch class 0..11, C = 0)
# ----------------------------------------------------------------------------
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ----------------------------------------------------------------------------
# Hilbert curve indexing (Wikipedia xy<->d, no dependency)
# ----------------------------------------------------------------------------
def hilbert_d2xy(n, d):
    """Map distance d along a Hilbert curve of side n (power of 2) to (x, y)."""
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        # rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_order(side):
    """Return arrays (xs, ys) giving the Hilbert traversal order of a side x side grid."""
    total = side * side
    xs = np.empty(total, dtype=np.int64)
    ys = np.empty(total, dtype=np.int64)
    for d in range(total):
        x, y = hilbert_d2xy(side, d)
        xs[d] = x
        ys[d] = y
    return xs, ys


# Cache Hilbert orders per side (they never change)
_HILBERT_CACHE = {}


def get_hilbert_order(side):
    if side not in _HILBERT_CACHE:
        _HILBERT_CACHE[side] = hilbert_order(side)
    return _HILBERT_CACHE[side]


# ----------------------------------------------------------------------------
# Color -> HSV -> pitch class
# ----------------------------------------------------------------------------
def rgb_grid_to_hsv(rgb):
    """rgb: (side, side, 3) uint8 -> h[0..1), s[0..1], v[0..1] float arrays."""
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    # avoid div by zero
    nz = diff > 1e-9
    # red is max
    rmask = nz & (mx == r)
    gmask = nz & (mx == g)
    bmask = nz & (mx == b)
    h[rmask] = ((g[rmask] - b[rmask]) / diff[rmask]) % 6
    h[gmask] = ((b[gmask] - r[gmask]) / diff[gmask]) + 2
    h[bmask] = ((r[bmask] - g[bmask]) / diff[bmask]) + 4
    h = h / 6.0  # 0..1

    s = np.where(mx > 1e-9, diff / np.where(mx > 1e-9, mx, 1), 0.0)
    v = mx
    return h, s, v


# ----------------------------------------------------------------------------
# Pitch-class SET THEORY (Forte / Rahn)
# ----------------------------------------------------------------------------
def _rotations_compact(pcs):
    """Given a sorted list of distinct pcs, return the most-compact rotation (normal order)."""
    n = len(pcs)
    if n == 0:
        return ((), [], [])
    if n == 1:
        return ((0,), [pcs[0]], [0])
    best = None
    for i in range(n):
        rot = [(pcs[(i + j) % n] - pcs[i]) % 12 for j in range(n)]
        # compare by span, then by left-packing (Rahn)
        # store the actual rotation (absolute, starting at pcs[i]) for normal order
        absrot = [pcs[(i + j) % n] for j in range(n)]
        key = _compactness_key(rot)
        if best is None or key < best[0]:
            best = (key, absrot, rot)
    return best  # (key, absolute normal order, intervals-from-first)


def _compactness_key(intervals_from_first):
    """Rahn compactness: smallest span; tie -> smallest at successively earlier positions."""
    # intervals_from_first[-1] is total span; then compare interior packing left to right (largest first)
    span = intervals_from_first[-1]
    # for ties, Rahn compares from the outside in; we approximate with the full tuple reversed
    return (span,) + tuple(reversed(intervals_from_first[:-1]))


def normal_order(pc_set):
    pcs = sorted(set(int(p) % 12 for p in pc_set))
    if not pcs:
        return []
    res = _rotations_compact(pcs)
    return res[1]


def prime_form(pc_set):
    """Forte prime form via Rahn: best of normal order of set and its inversion, transposed to 0."""
    pcs = sorted(set(int(p) % 12 for p in pc_set))
    if not pcs:
        return []

    def t0(no):
        base = no[0]
        return [(p - base) % 12 for p in no]

    candidates = []
    for s in (pcs, sorted((12 - p) % 12 for p in pcs)):
        res = _rotations_compact(s)
        candidates.append(t0(res[1]))

    # choose the more left-packed
    def packing_key(c):
        return tuple(c)

    return min(candidates, key=packing_key)


def interval_class_vector(pc_set):
    pcs = sorted(set(int(p) % 12 for p in pc_set))
    icv = [0, 0, 0, 0, 0, 0]
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            ic = (pcs[j] - pcs[i]) % 12
            if ic > 6:
                ic = 12 - ic
            if 1 <= ic <= 6:
                icv[ic - 1] += 1
    return icv


# Complete set-class table (all 224 classes), keyed by Rahn prime form.
# Generated from music21; A/B suffix marks inversionally-related pairs.
FORTE_TABLE = {
    (): '0-1',
    (0,): '1-1',
    (0, 1): '2-1',
    (0, 2): '2-2',
    (0, 3): '2-3',
    (0, 4): '2-4',
    (0, 5): '2-5',
    (0, 6): '2-6',
    (0, 1, 2): '3-1',
    (0, 1, 3): '3-2A',
    (0, 1, 4): '3-3A',
    (0, 1, 5): '3-4A',
    (0, 1, 6): '3-5A',
    (0, 2, 4): '3-6',
    (0, 2, 5): '3-7A',
    (0, 2, 6): '3-8A',
    (0, 2, 7): '3-9',
    (0, 3, 6): '3-10',
    (0, 3, 7): '3-11A',
    (0, 4, 8): '3-12',
    (0, 1, 2, 3): '4-1',
    (0, 1, 2, 4): '4-2A',
    (0, 1, 2, 5): '4-4A',
    (0, 1, 2, 6): '4-5A',
    (0, 1, 2, 7): '4-6',
    (0, 1, 3, 4): '4-3',
    (0, 1, 3, 5): '4-11A',
    (0, 1, 3, 6): '4-13A',
    (0, 1, 3, 7): '4-29A',
    (0, 1, 4, 5): '4-7',
    (0, 1, 4, 6): '4-15A',
    (0, 1, 4, 7): '4-18A',
    (0, 1, 4, 8): '4-19A',
    (0, 1, 5, 6): '4-8',
    (0, 1, 5, 7): '4-16A',
    (0, 1, 5, 8): '4-20',
    (0, 1, 6, 7): '4-9',
    (0, 2, 3, 5): '4-10',
    (0, 2, 3, 6): '4-12B',
    (0, 2, 3, 7): '4-14B',
    (0, 2, 4, 6): '4-21',
    (0, 2, 4, 7): '4-22A',
    (0, 2, 4, 8): '4-24',
    (0, 2, 5, 7): '4-23',
    (0, 2, 5, 8): '4-27A',
    (0, 2, 6, 8): '4-25',
    (0, 3, 4, 7): '4-17',
    (0, 3, 5, 8): '4-26',
    (0, 3, 6, 9): '4-28',
    (0, 1, 2, 3, 4): '5-1',
    (0, 1, 2, 3, 5): '5-2A',
    (0, 1, 2, 3, 6): '5-4A',
    (0, 1, 2, 3, 7): '5-5A',
    (0, 1, 2, 4, 5): '5-3A',
    (0, 1, 2, 4, 6): '5-9A',
    (0, 1, 2, 4, 7): '5-36A',
    (0, 1, 2, 4, 8): '5-13A',
    (0, 1, 2, 5, 6): '5-6A',
    (0, 1, 2, 5, 7): '5-14A',
    (0, 1, 2, 5, 8): '5-38A',
    (0, 1, 2, 6, 7): '5-7A',
    (0, 1, 2, 6, 8): '5-15',
    (0, 1, 3, 4, 6): '5-10A',
    (0, 1, 3, 4, 7): '5-16A',
    (0, 1, 3, 4, 8): '5-17',
    (0, 1, 3, 5, 6): '5-12',
    (0, 1, 3, 5, 7): '5-24A',
    (0, 1, 3, 5, 8): '5-27A',
    (0, 1, 3, 6, 7): '5-19A',
    (0, 1, 3, 6, 8): '5-29A',
    (0, 1, 3, 6, 9): '5-31A',
    (0, 1, 4, 5, 7): '5-18A',
    (0, 1, 4, 5, 8): '5-21A',
    (0, 1, 4, 6, 8): '5-30A',
    (0, 1, 4, 6, 9): '5-32A',
    (0, 1, 4, 7, 8): '5-22',
    (0, 1, 5, 6, 8): '5-20A',
    (0, 2, 3, 4, 6): '5-8',
    (0, 2, 3, 4, 7): '5-11B',
    (0, 2, 3, 5, 7): '5-23A',
    (0, 2, 3, 5, 8): '5-25A',
    (0, 2, 3, 6, 8): '5-28B',
    (0, 2, 4, 5, 8): '5-26B',
    (0, 2, 4, 6, 8): '5-33',
    (0, 2, 4, 6, 9): '5-34',
    (0, 2, 4, 7, 9): '5-35',
    (0, 3, 4, 5, 8): '5-37',
    (0, 1, 2, 3, 4, 5): '6-1',
    (0, 1, 2, 3, 4, 6): '6-2A',
    (0, 1, 2, 3, 4, 7): '6-36A',
    (0, 1, 2, 3, 4, 8): '6-37',
    (0, 1, 2, 3, 5, 6): '6-3A',
    (0, 1, 2, 3, 5, 7): '6-9A',
    (0, 1, 2, 3, 5, 8): '6-40A',
    (0, 1, 2, 3, 6, 7): '6-5A',
    (0, 1, 2, 3, 6, 8): '6-41A',
    (0, 1, 2, 3, 6, 9): '6-42',
    (0, 1, 2, 3, 7, 8): '6-38',
    (0, 1, 2, 4, 5, 6): '6-4',
    (0, 1, 2, 4, 5, 7): '6-11A',
    (0, 1, 2, 4, 5, 8): '6-15A',
    (0, 1, 2, 4, 6, 7): '6-12A',
    (0, 1, 2, 4, 6, 8): '6-22A',
    (0, 1, 2, 4, 6, 9): '6-46A',
    (0, 1, 2, 4, 7, 8): '6-17A',
    (0, 1, 2, 4, 7, 9): '6-47A',
    (0, 1, 2, 5, 6, 7): '6-6',
    (0, 1, 2, 5, 6, 8): '6-43A',
    (0, 1, 2, 5, 6, 9): '6-44A',
    (0, 1, 2, 5, 7, 8): '6-18A',
    (0, 1, 2, 5, 7, 9): '6-48',
    (0, 1, 2, 6, 7, 8): '6-7',
    (0, 1, 3, 4, 5, 7): '6-10B',
    (0, 1, 3, 4, 5, 8): '6-14B',
    (0, 1, 3, 4, 6, 7): '6-13',
    (0, 1, 3, 4, 6, 8): '6-24A',
    (0, 1, 3, 4, 6, 9): '6-27A',
    (0, 1, 3, 4, 7, 8): '6-19A',
    (0, 1, 3, 4, 7, 9): '6-49',
    (0, 1, 3, 5, 6, 8): '6-25A',
    (0, 1, 3, 5, 6, 9): '6-28',
    (0, 1, 3, 5, 7, 8): '6-26',
    (0, 1, 3, 5, 7, 9): '6-34A',
    (0, 1, 3, 6, 7, 9): '6-30A',
    (0, 1, 4, 5, 6, 8): '6-16A',
    (0, 1, 4, 5, 7, 9): '6-31A',
    (0, 1, 4, 5, 8, 9): '6-20',
    (0, 1, 4, 6, 7, 9): '6-50',
    (0, 2, 3, 4, 5, 7): '6-8',
    (0, 2, 3, 4, 5, 8): '6-39B',
    (0, 2, 3, 4, 6, 8): '6-21A',
    (0, 2, 3, 4, 6, 9): '6-45',
    (0, 2, 3, 5, 6, 8): '6-23',
    (0, 2, 3, 5, 7, 9): '6-33A',
    (0, 2, 3, 6, 7, 9): '6-29',
    (0, 2, 4, 5, 7, 9): '6-32',
    (0, 2, 4, 6, 8, 10): '6-35',
    (0, 1, 2, 3, 4, 5, 6): '7-1',
    (0, 1, 2, 3, 4, 5, 7): '7-2A',
    (0, 1, 2, 3, 4, 5, 8): '7-3A',
    (0, 1, 2, 3, 4, 6, 7): '7-4A',
    (0, 1, 2, 3, 4, 6, 8): '7-9A',
    (0, 1, 2, 3, 4, 6, 9): '7-10A',
    (0, 1, 2, 3, 4, 7, 8): '7-6A',
    (0, 1, 2, 3, 4, 7, 9): '7-12',
    (0, 1, 2, 3, 5, 6, 7): '7-5A',
    (0, 1, 2, 3, 5, 6, 8): '7-36A',
    (0, 1, 2, 3, 5, 6, 9): '7-16A',
    (0, 1, 2, 3, 5, 7, 8): '7-14A',
    (0, 1, 2, 3, 5, 7, 9): '7-24A',
    (0, 1, 2, 3, 6, 7, 8): '7-7A',
    (0, 1, 2, 3, 6, 7, 9): '7-19A',
    (0, 1, 2, 4, 5, 6, 8): '7-13A',
    (0, 1, 2, 4, 5, 6, 9): '7-17',
    (0, 1, 2, 4, 5, 7, 8): '7-38A',
    (0, 1, 2, 4, 5, 7, 9): '7-27A',
    (0, 1, 2, 4, 5, 8, 9): '7-21A',
    (0, 1, 2, 4, 6, 7, 8): '7-15',
    (0, 1, 2, 4, 6, 7, 9): '7-29A',
    (0, 1, 2, 4, 6, 8, 9): '7-30A',
    (0, 1, 2, 4, 6, 8, 10): '7-33',
    (0, 1, 2, 5, 6, 7, 9): '7-20A',
    (0, 1, 2, 5, 6, 8, 9): '7-22',
    (0, 1, 3, 4, 5, 6, 8): '7-11B',
    (0, 1, 3, 4, 5, 7, 8): '7-37',
    (0, 1, 3, 4, 5, 7, 9): '7-26B',
    (0, 1, 3, 4, 6, 7, 9): '7-31A',
    (0, 1, 3, 4, 6, 8, 9): '7-32A',
    (0, 1, 3, 4, 6, 8, 10): '7-34',
    (0, 1, 3, 5, 6, 7, 9): '7-28B',
    (0, 1, 3, 5, 6, 8, 10): '7-35',
    (0, 1, 4, 5, 6, 7, 9): '7-18A',
    (0, 2, 3, 4, 5, 6, 8): '7-8',
    (0, 2, 3, 4, 5, 7, 9): '7-23A',
    (0, 2, 3, 4, 6, 7, 9): '7-25A',
    (0, 1, 2, 3, 4, 5, 6, 7): '8-1',
    (0, 1, 2, 3, 4, 5, 6, 8): '8-2A',
    (0, 1, 2, 3, 4, 5, 6, 9): '8-3',
    (0, 1, 2, 3, 4, 5, 7, 8): '8-4A',
    (0, 1, 2, 3, 4, 5, 7, 9): '8-11A',
    (0, 1, 2, 3, 4, 5, 8, 9): '8-7',
    (0, 1, 2, 3, 4, 6, 7, 8): '8-5A',
    (0, 1, 2, 3, 4, 6, 7, 9): '8-13A',
    (0, 1, 2, 3, 4, 6, 8, 9): '8-15A',
    (0, 1, 2, 3, 4, 6, 8, 10): '8-21',
    (0, 1, 2, 3, 4, 7, 8, 9): '8-8',
    (0, 1, 2, 3, 5, 6, 7, 8): '8-6',
    (0, 1, 2, 3, 5, 6, 7, 9): '8-29A',
    (0, 1, 2, 3, 5, 6, 8, 9): '8-18A',
    (0, 1, 2, 3, 5, 6, 8, 10): '8-22A',
    (0, 1, 2, 3, 5, 7, 8, 9): '8-16A',
    (0, 1, 2, 3, 5, 7, 8, 10): '8-23',
    (0, 1, 2, 3, 6, 7, 8, 9): '8-9',
    (0, 1, 2, 4, 5, 6, 7, 9): '8-14B',
    (0, 1, 2, 4, 5, 6, 8, 9): '8-19A',
    (0, 1, 2, 4, 5, 6, 8, 10): '8-24',
    (0, 1, 2, 4, 5, 7, 8, 9): '8-20',
    (0, 1, 2, 4, 5, 7, 8, 10): '8-27A',
    (0, 1, 2, 4, 6, 7, 8, 10): '8-25',
    (0, 1, 3, 4, 5, 6, 7, 9): '8-12B',
    (0, 1, 3, 4, 5, 6, 8, 9): '8-17',
    (0, 1, 3, 4, 5, 7, 8, 10): '8-26',
    (0, 1, 3, 4, 6, 7, 9, 10): '8-28',
    (0, 2, 3, 4, 5, 6, 7, 9): '8-10',
    (0, 1, 2, 3, 4, 5, 6, 7, 8): '9-1',
    (0, 1, 2, 3, 4, 5, 6, 7, 9): '9-2A',
    (0, 1, 2, 3, 4, 5, 6, 8, 9): '9-3A',
    (0, 1, 2, 3, 4, 5, 6, 8, 10): '9-6',
    (0, 1, 2, 3, 4, 5, 7, 8, 9): '9-4A',
    (0, 1, 2, 3, 4, 5, 7, 8, 10): '9-7A',
    (0, 1, 2, 3, 4, 6, 7, 8, 9): '9-5A',
    (0, 1, 2, 3, 4, 6, 7, 8, 10): '9-8A',
    (0, 1, 2, 3, 4, 6, 7, 9, 10): '9-10',
    (0, 1, 2, 3, 5, 6, 7, 8, 10): '9-9',
    (0, 1, 2, 3, 5, 6, 7, 9, 10): '9-11A',
    (0, 1, 2, 4, 5, 6, 8, 9, 10): '9-12',
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9): '10-1',
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 10): '10-2',
    (0, 1, 2, 3, 4, 5, 6, 7, 9, 10): '10-3',
    (0, 1, 2, 3, 4, 5, 6, 8, 9, 10): '10-4',
    (0, 1, 2, 3, 4, 5, 7, 8, 9, 10): '10-5',
    (0, 1, 2, 3, 4, 6, 7, 8, 9, 10): '10-6',
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10): '11-1',
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11): '12-1',
}

# Plain-language names for landmark sets (by prime form)
SET_NICKNAMES = {
    (0, 3, 7): "major/minor triad",
    (0, 4, 8): "augmented triad",
    (0, 3, 6): "diminished triad",
    (0, 2, 7): "suspended / quartal trio",
    (0, 2, 4, 7): "added-sixth / pentatonic fragment",
    (0, 2, 5, 7): "quartal tetrad",
    (0, 3, 6, 9): "diminished seventh",
    (0, 2, 4, 6, 8): "whole-tone pentad",
    (0, 2, 4, 7, 9): "major pentatonic",
    (0, 2, 4, 6, 8, 10): "whole-tone scale",
    (0, 1, 3, 5, 7, 9): "Guidonian / odd whole-tone",
    (0, 2, 3, 5, 7, 9): "Dorian hexachord",
    (0, 1, 3, 4, 6, 7, 9, 10): "octatonic scale",
    (0, 1, 3, 5, 6, 8, 10): "major scale (diatonic)",
    (0, 2, 4, 5, 7, 9, 11): "major scale (diatonic)",
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11): "chromatic aggregate",
}


def forte_number(pc_set):
    pf = tuple(prime_form(pc_set))
    return FORTE_TABLE.get(pf, "?")


def set_nickname(pc_set):
    pf = tuple(prime_form(pc_set))
    return SET_NICKNAMES.get(pf, "")


def analyze_set(pc_set):
    pcs = sorted(set(int(p) % 12 for p in pc_set))
    no = normal_order(pcs)
    pf = prime_form(pcs)
    return {
        "pcs": pcs,
        "pc_names": [PC_NAMES[p] for p in pcs],
        "normal_order": no,
        "prime_form": pf,
        "forte": FORTE_TABLE.get(tuple(pf), "?"),
        "icv": interval_class_vector(pcs),
        "nickname": SET_NICKNAMES.get(tuple(pf), ""),
        "cardinality": len(pcs),
    }


# ----------------------------------------------------------------------------
# Ingest: every pixel -> pooled, blurred 2^k Hilbert grid
# ----------------------------------------------------------------------------
def ingest_image(img, k=8, blur=1.0):
    side = 2 ** k
    img = img.convert("RGB")
    orig_px = img.width * img.height
    if blur and blur > 0:
        # blur in original resolution so neighbors bleed before pooling
        img = img.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    # BOX resampling area-averages EVERY original pixel into its cell
    grid = img.resize((side, side), Image.BOX)
    rgb = np.asarray(grid, dtype=np.uint8)  # (side, side, 3)
    return rgb, side, orig_px


# ----------------------------------------------------------------------------
# Overlapping windows along the Hilbert stream -> per-segment pitch-class sets
# ----------------------------------------------------------------------------
def windowed_sets(rgb, side, n_events, overlap=1.6, threshold=0.03):
    """Return list of per-event dicts: pcs set, weights, mean value/sat, top pc."""
    h, s, v = rgb_grid_to_hsv(rgb)
    # hue -> pitch class
    pc = np.round(h * 12.0).astype(np.int64) % 12  # (side, side)

    xs, ys = get_hilbert_order(side)
    pc_seq = pc[ys, xs]          # 1-D along Hilbert curve
    s_seq = s[ys, xs]
    v_seq = v[ys, xs]
    total = pc_seq.shape[0]

    base = total / n_events
    half = (base * overlap) / 2.0

    events = []
    for i in range(n_events):
        center = (i + 0.5) * base
        lo = max(0, int(center - half))
        hi = min(total, int(center + half))
        if hi <= lo:
            hi = min(total, lo + 1)
        wpc = pc_seq[lo:hi]
        ws = s_seq[lo:hi]
        wv = v_seq[lo:hi]
        # saturation-weighted pitch-class histogram (grays barely vote)
        weights = ws + 0.05  # floor so fully-gray windows still pick a tonic
        hist = np.bincount(wpc, weights=weights, minlength=12)
        tot = hist.sum()
        if tot <= 0:
            hist = np.bincount(wpc, minlength=12).astype(float)
            tot = max(hist.sum(), 1.0)
        frac = hist / tot
        # threshold rare pcs, but always keep at least the top one
        keep = np.where(frac >= threshold)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(frac))])
        top = int(np.argmax(frac))
        events.append({
            "pcs": sorted(int(p) for p in keep),
            "weights": {int(p): float(frac[p]) for p in keep},
            "top": top,
            "mean_val": float(wv.mean()),
            "mean_sat": float(ws.mean()),
        })
    return events


# ----------------------------------------------------------------------------
# Compose: per-event pitch-class sets -> voiced notes
# ----------------------------------------------------------------------------
# Diatonic snap table for "tonal" mode: nearest pc in C major-ish set centered on tonic
def _tonal_snap(pc, tonic):
    scale = [0, 2, 4, 5, 7, 9, 11]  # major scale degrees
    rel = (pc - tonic) % 12
    if rel in scale:
        return pc
    # snap to nearest scale degree
    best = min(scale, key=lambda d: min((rel - d) % 12, (d - rel) % 12))
    return (tonic + best) % 12


def compose(events, duration, mode="faithful", seed=0):
    """Return dict with tempo, events[], and a flat note list for playback/MIDI."""
    n = len(events)
    sec_per_event = duration / n
    rng = np.random.default_rng(seed)

    # tonic = most common 'top' across events (a tonal center the piece can lean on)
    tops = np.bincount([e["top"] for e in events], minlength=12)
    tonic = int(np.argmax(tops))

    notes = []  # {t, dur, midi, vel, voice}
    out_events = []
    prev_melody = None

    for i, e in enumerate(events):
        t = i * sec_per_event
        pcs = list(e["pcs"])
        if mode == "tonal":
            pcs = sorted(set(_tonal_snap(p, tonic) for p in pcs))

        mean_v = e["mean_val"]
        mean_s = e["mean_sat"]

        # register from lightness: dark -> low, bright -> high
        chord_oct = 3 + int(round(mean_v * 2))      # 3..5
        bass_oct = 2
        mel_oct = 4 + int(round(mean_v * 2))        # 4..6
        vel = int(40 + mean_s * 70)                 # 40..110

        # --- pad / chord voice: the whole set sounds ---
        chord_pcs = pcs[:6]
        for p in chord_pcs:
            midi = 12 * (chord_oct + 1) + p
            notes.append({"t": round(t, 4), "dur": round(sec_per_event * 0.95, 4),
                          "midi": int(midi), "vel": int(vel * 0.6), "voice": "pad"})

        # --- bass: root of the set (top pc), slow ---
        root = e["top"] if mode == "faithful" else _tonal_snap(e["top"], tonic)
        bass_midi = 12 * (bass_oct + 1) + root
        notes.append({"t": round(t, 4), "dur": round(sec_per_event * 0.95, 4),
                      "midi": int(bass_midi), "vel": int(vel * 0.7), "voice": "bass"})

        # --- melody: pick a pc weighted toward the most-prevalent, with voice-leading ---
        weights = e["weights"]
        if mode == "tonal" and weights:
            weights = {}
            for p in pcs:
                weights[p] = e["weights"].get(p, 0.0) + 0.001
        choice_pcs = list(weights.keys()) if weights else pcs
        probs = np.array([weights.get(p, 0.001) for p in choice_pcs], dtype=float)
        probs = probs / probs.sum()
        mel_pc = int(rng.choice(choice_pcs, p=probs))
        mel_midi = 12 * (mel_oct + 1) + mel_pc
        # voice-leading: nudge toward previous melody note within an octave
        if prev_melody is not None:
            while mel_midi - prev_melody > 7:
                mel_midi -= 12
            while prev_melody - mel_midi > 7:
                mel_midi += 12
        prev_melody = mel_midi
        # melody is a touch faster: two hits per event when the patch is vivid
        mel_dur = sec_per_event * (0.5 if mean_s > 0.4 else 0.95)
        notes.append({"t": round(t, 4), "dur": round(mel_dur, 4),
                      "midi": int(mel_midi), "vel": int(min(vel + 10, 120)), "voice": "melody"})
        if mean_s > 0.4:
            notes.append({"t": round(t + sec_per_event * 0.5, 4), "dur": round(sec_per_event * 0.45, 4),
                          "midi": int(mel_midi), "vel": int(min(vel, 110)), "voice": "melody"})

        out_events.append({
            "i": i, "t": round(t, 4), "pcs": pcs,
            "pc_names": [PC_NAMES[p] for p in pcs],
            "top": e["top"], "mean_val": round(mean_v, 3), "mean_sat": round(mean_s, 3),
        })

    return {
        "tonic": tonic,
        "tonic_name": PC_NAMES[tonic],
        "sec_per_event": round(sec_per_event, 4),
        "n_events": n,
        "duration": duration,
        "events": out_events,
        "notes": notes,
    }


# ----------------------------------------------------------------------------
# Minimal Standard MIDI File writer (no dependency)
# ----------------------------------------------------------------------------
def _vlq(n):
    """Variable-length quantity encoding."""
    buf = n & 0x7F
    n >>= 7
    out = bytearray()
    while n:
        buf <<= 8
        buf |= ((n & 0x7F) | 0x80)
        n >>= 7
    while True:
        out.append(buf & 0xFF)
        if buf & 0x80:
            buf >>= 8
        else:
            break
    return bytes(out)


def write_midi(notes, duration, path, ticks_per_beat=480, bpm=120):
    """One-track SMF from absolute-time notes (seconds)."""
    sec_per_beat = 60.0 / bpm
    tps = ticks_per_beat / sec_per_beat  # ticks per second

    # build (tick, status, data1, data2) events
    raw = []
    for nt in notes:
        on = int(round(nt["t"] * tps))
        off = int(round((nt["t"] + max(nt["dur"], 0.02)) * tps))
        ch = {"pad": 0, "bass": 1, "melody": 2}.get(nt["voice"], 0)
        m = max(0, min(127, nt["midi"]))
        raw.append((on, 0x90 | ch, m, max(1, min(127, nt["vel"]))))
        raw.append((off, 0x80 | ch, m, 0))
    raw.sort(key=lambda x: x[0])

    track = bytearray()
    # tempo meta
    mpqn = int(60_000_000 / bpm)
    track += _vlq(0) + b"\xFF\x51\x03" + struct.pack(">I", mpqn)[1:]
    # program changes: pad=warm pad(89), bass=synth bass(38), melody=lead(80)
    for ch, prog in ((0, 89), (1, 38), (2, 80)):
        track += _vlq(0) + bytes([0xC0 | ch, prog])

    last = 0
    for tick, status, d1, d2 in raw:
        delta = tick - last
        last = tick
        track += _vlq(delta) + bytes([status, d1, d2])
    # end of track
    end_tick = int(round(duration * tps)) + ticks_per_beat
    track += _vlq(max(0, end_tick - last)) + b"\xFF\x2F\x00"

    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat))
        f.write(b"MTrk" + struct.pack(">I", len(track)) + track)


# ----------------------------------------------------------------------------
# Audio render -> WAV (numpy additive synth; plays in any media player, no SoundFont)
# ----------------------------------------------------------------------------
import wave

# per-voice harmonic series (amplitude of each partial) and ADSR + gain
VOICE_TIMBRE = {
    "pad":    {"harm": [1.0, 0.35, 0.15, 0.05], "adsr": (0.40, 0.30, 0.6, 0.8), "gain": 0.5},
    "bass":   {"harm": [1.0, 0.0, 0.18, 0.0, 0.06], "adsr": (0.05, 0.20, 0.7, 0.4), "gain": 0.85},
    "melody": {"harm": [1.0, 0.0, 0.28, 0.0, 0.12, 0.0, 0.06], "adsr": (0.02, 0.15, 0.45, 0.3), "gain": 0.7},
}


def _adsr_env(length, sr, a, d, s, r):
    env = np.empty(length, np.float32)
    A = min(int(a * sr), length)
    R = min(int(r * sr), length - A)
    D = min(int(d * sr), length - A - R)
    S = length - A - D - R
    i = 0
    if A: env[i:i + A] = np.linspace(0.0, 1.0, A, endpoint=False); i += A
    if D: env[i:i + D] = np.linspace(1.0, s, D, endpoint=False); i += D
    if S: env[i:i + S] = s; i += S
    if R: env[i:i + R] = np.linspace(s, 0.0, R, endpoint=True); i += R
    if i < length: env[i:] = 0.0
    return env


def synth_wav(comp, path, sr=44100):
    notes = comp["notes"]
    duration = comp["duration"]
    total = int((duration + 2.5) * sr)
    buf = np.zeros(total, np.float32)
    for n in notes:
        f = 440.0 * 2 ** ((n["midi"] - 69) / 12.0)
        start = int(n["t"] * sr)
        length = int(max(n["dur"], 0.05) * sr)
        if start >= total: continue
        length = min(length, total - start)
        if length <= 0: continue
        tim = VOICE_TIMBRE.get(n["voice"], VOICE_TIMBRE["pad"])
        t = np.arange(length, dtype=np.float32) / sr
        wave_arr = np.zeros(length, np.float32)
        for k, amp in enumerate(tim["harm"], start=1):
            if amp == 0.0: continue
            if f * k > sr * 0.45: break          # avoid aliasing
            wave_arr += amp * np.sin(2 * np.pi * f * k * t)
        env = _adsr_env(length, sr, *tim["adsr"])
        amp = (n["vel"] / 127.0) * tim["gain"]
        buf[start:start + length] += wave_arr * env * amp

    # light stereo-ish space: a couple of attenuated echoes
    echo = np.zeros_like(buf)
    for delay, g in ((0.11, 0.22), (0.23, 0.12)):
        ds = int(delay * sr)
        if ds < total: echo[ds:] += buf[:total - ds] * g
    buf = buf + echo

    # normalize + gentle soft clip
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = np.tanh(buf / peak * 1.1) * 0.92
    pcm = (buf * 32767).astype("<i2")

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# Last composition (for lazy WAV render on demand, so /api/analyze stays fast)
LAST_COMPOSITION = None


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no image"}), 400
    try:
        img = Image.open(io.BytesIO(f.read()))
    except Exception as ex:
        return jsonify({"error": f"could not read image: {ex}"}), 400

    k = int(request.form.get("detail", 8))
    k = max(5, min(9, k))
    blur = float(request.form.get("blur", 1.0))
    duration = float(request.form.get("duration", 120))
    duration = max(20, min(600, duration))
    eps = float(request.form.get("events_per_sec", 4))
    eps = max(0.5, min(12, eps))
    overlap = float(request.form.get("overlap", 1.6))
    threshold = float(request.form.get("threshold", 0.03))
    mode = request.form.get("mode", "faithful")
    seed = int(request.form.get("seed", 0))

    n_events = max(4, int(round(duration * eps)))

    rgb, side, orig_px = ingest_image(img, k=k, blur=blur)
    events = windowed_sets(rgb, side, n_events, overlap=overlap, threshold=threshold)

    # global set = all pcs that appear across events
    all_pcs = sorted(set(p for e in events for p in e["pcs"]))
    global_analysis = analyze_set(all_pcs)

    comp = compose(events, duration, mode=mode, seed=seed)

    # write MIDI
    midi_name = "chromatic-set.mid"
    write_midi(comp["notes"], duration, os.path.join(OUT_DIR, midi_name))

    # stash for lazy WAV render (keeps compose fast; WAV built only when requested)
    global LAST_COMPOSITION
    LAST_COMPOSITION = comp

    # grid preview (small PNG, base64) so the user sees the pooled/blurred field
    preview = Image.fromarray(rgb, "RGB").resize((256, 256), Image.NEAREST)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # per-event swatch hex (mean color of each event window, for the timeline)
    return jsonify({
        "ok": True,
        "orig_pixels": orig_px,
        "grid_side": side,
        "per_pixel_ms": round(duration * 1000.0 / max(orig_px, 1), 6),
        "global": global_analysis,
        "composition": comp,
        "midi_url": f"/api/midi?name={midi_name}",
        "wav_url": "/api/wav",
        "preview_png": preview_b64,
        "params": {
            "detail": k, "blur": blur, "duration": duration,
            "events_per_sec": eps, "overlap": overlap,
            "threshold": threshold, "mode": mode, "n_events": n_events,
        },
    })


@app.route("/api/midi")
def api_midi():
    name = request.args.get("name", "chromatic-set.mid")
    name = os.path.basename(name)
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="audio/midi", as_attachment=True,
                     download_name="chromatic-set.mid")


@app.route("/api/wav")
def api_wav():
    if LAST_COMPOSITION is None:
        return jsonify({"error": "compose something first"}), 400
    path = os.path.join(OUT_DIR, "chromatic-set.wav")
    synth_wav(LAST_COMPOSITION, path)
    return send_file(path, mimetype="audio/wav", as_attachment=True,
                     download_name="chromatic-set.wav")


if __name__ == "__main__":
    print(f"Chromatic Set running at http://localhost:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
