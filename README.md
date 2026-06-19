# Chromatic Set

Upload an image. Every pixel's color is read, pooled with its neighbors, and turned into a
sequence of **pitch-class sets** (real Forte set theory) that becomes the basis of a composition
about two minutes long.

## The idea

The **hue wheel (0-360°) is a circle of twelve, just like the chromatic pitch circle**, so a color
maps onto a pitch class almost natively (red = C by default). The app uses that to read a whole
image as music.

A pixel never becomes its own note (a megapixel image would be thousands of notes per second).
Instead each pixel influences the piece through **three layers of neighbor interaction**:

1. **Area-pool** — the full image is box-resized to a `2^k × 2^k` grid (default 256×256). Box
   resampling area-averages *every* original pixel into its cell. Nothing is discarded.
2. **Neighbor blend** — a Gaussian blur lets cells bleed into adjacent cells before sonifying.
3. **Hilbert + overlapping windows** — grid cells are walked along a Hilbert curve (so spatial
   neighbors stay temporal neighbors), and each musical event's pitch-class set is a
   saturation-weighted histogram over a window of cells that overlaps its neighbors. Rare pitch
   classes (below a threshold) are dropped so a noisy patch can't force a 12-note cluster.

**Note rate is a musical dial** (events/sec), decoupled from pixel count. Pixel count controls
*smoothness*: a 50MP photo and a thumbnail both yield a ~2-minute piece; the big one just sounds
smoother because more pixels average into each set.

For each event the set's pitches are voiced as a pad chord + bass root + melody line. Lightness
sets register, saturation sets dynamics. A **Faithful (atonal)** ↔ **Tonal** toggle changes only
the voicing, not the analysis.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
# open http://localhost:5018
```

Or build the launcher: `bash setup.sh` → `~/Desktop/Apps/Chromatic Set.app`.

## Controls

- **Character** — Faithful (atonal) vs Tonal (snaps to a scale).
- **Length** — target duration (~120s default).
- **Events / sec** — note density (the music dial; default 4).
- **Neighbor blend** — Gaussian radius; how much each pixel talks to its neighbors.
- **Detail (grid)** — Hilbert grid side `2^k`; bigger = finer harmonic detail.
- **Set threshold** — minimum share of a window a pitch class needs to be kept.

Output: in-browser playback (Tone.js), an animated chromatic clock + piano roll, and a
**MIDI download** that opens in any DAW or notation app.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask server (:5018): ingest, set-theory analysis, composition, MIDI |
| `web/index.html` | UI (HTML+CSS+JS), Tone.js playback, clock + roll |
| `setup.sh` | Builds the macOS `.app` bundle |
| `create_icon.py` | Generates the flat spectrum-disc icon |

## Gitignored (not in the repo)

- `uploads/`, `out/`, `*.mid`, `*.wav` — uploaded images and rendered output are personal data.
- `.venv/`, `__pycache__/`, `.DS_Store`.

To rebuild after cloning: create the venv and install `requirements.txt` as above. The `uploads/`
and `out/` directories are recreated automatically at runtime.
