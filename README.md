# how_many

how_many is a small, always-on-top desktop overlay that helps you count repeated
structures directly on top of any image (semiconductor dies, industrial
patterns, biological microscopy, and more). You position a draggable line across
the features you want to measure and the tool analyses the stripe underneath the
line to suggest how many items are present.

## Feature highlights

- Transparent overlay window with two draggable endpoints and tick marks aligned
  to a configurable stripe beneath the line.
- Control dialog containing analysis controls, profile visualisation, and a help
  reference.
- Automatic suggestion of likely item counts using FFT and autocorrelation, with
  explicit correction for the “items = cycles + 1” off-by-one that many tools
  miss.
- Manual override for the item count, with immediate overlay feedback.
- Profile tab that displays the 1-D signal extracted from the stripe as well as
  the candidate list used to populate the UI.
- Keyboard shortcuts for the most common adjustments (analysis trigger, stripe
  width, manual counts, closing the overlay).
- Persisted configuration at `~/.how_many_config.json` so your preferred
  settings travel between runs.

## User interface overview

### Overlay window

The overlay stays always on top and exposes a line segment with two endpoints.
Drag either endpoint or the entire line to reposition it over the features you
want to count. Holding `Ctrl` while dragging an endpoint snaps the angle to
0°/45°/90°/135°. Tick markers are spaced so that the endpoints are always
included, making it easy to visually confirm the count. A heads-up display shows
line length `L`, stripe width `W`, and the active item count `N` offset slightly
from the stripe so it does not obstruct the view.

### Control dialog

The control dialog contains three tabs:

- **Controls** – Estimated counts dropdown with confidence and source, manual
  spin box for the item count, buttons for analysing now or enabling
  auto-analysis, stripe width and line style controls, and an “always on top”
  toggle.
- **Profile** – Graph of the 1-D intensity profile extracted from the stripe. It
  renders vertical markers at the inferred item boundaries and repeats the list
  of candidates for quick inspection.
- **Help** – Quick-reference of usage tips and keyboard shortcuts.

Keyboard shortcuts:

| Shortcut | Action                     |
| -------- | -------------------------- |
| `A`      | Trigger an analysis        |
| `+` / `-`| Increase/decrease item `N` |
| `W` / `Q`| Increase/decrease stripe `W` |
| `Esc`    | Close the overlay          |

## Analysis pipeline

1. **Capture** – The app briefly hides its windows (configurable via
   `hide_during_capture_ms`, default 40 ms) and grabs the desktop image.
2. **Stripe extraction** – The stripe under the line is mapped into the captured
   image and rotated so the line is horizontal. A band centred on the line is
   cropped (height = stripe width, width = line length).
3. **Profile** – The band is converted to grayscale and averaged vertically to
   produce a 1-D intensity profile vs. distance.
4. **Pre-processing** – The signal is detrended and windowed (Hanning) to reduce
   leakage.
5. **Periodicity detection** – FFT peaks and autocorrelation peaks are located,
   with simple harmonic heuristics. Each result is converted from cycles to
   items via the off-by-one fix (items = cycles + 1).
6. **Ranking** – Candidate counts from FFT, ACF, and harmonics are merged,
   confidence-ranked, and trimmed to the configured maximum suggestions (default
   5). The best candidate is applied automatically to the overlay.

## Repository layout

```
├── src/how_many          # Application package
│   ├── analysis          # FFT/ACF routines and standalone peak helpers
│   ├── utils             # Qt/image/geometry helper functions
│   ├── app.py            # Qt application entry point
│   ├── __init__.py       # Package export (exposes main())
│   └── __main__.py       # Enables `python -m how_many`
├── scripts/build_exe.py  # PyInstaller helper for local builds
├── tests/test_peaks_detrend.py  # Optional SciPy regression harness (skipped)
├── .github/workflows     # CI that builds the Windows executable
└── README.md / pyproject.toml   # Metadata and dependency declaration
```

The `src/how_many/analysis/peaks_detrend.py` module vendors pure-Python
implementations of `find_peaks` and `detrend` so the PyInstaller bundle does not
need SciPy. The optional regression harness in `tests/` can be run manually
against a SciPy checkout to verify behaviour parity.

## Building and running locally

```bash
# Install the project together with the build + test tooling
uv sync --group build --group test

# Launch the overlay from the managed environment
uv run python -m how_many

# Build a PyInstaller one-file executable for the current platform
uv run python scripts/build_exe.py
```

The PyInstaller command writes the platform-specific binary to `dist/` (for
example `how_many.exe` on Windows or `how_many` on Linux/macOS).

## Continuous integration

The GitHub Actions workflow (`.github/workflows/build.yml`) runs on every push
and pull request. It uses [uv](https://github.com/astral-sh/uv) to manage the
environment, executes the SciPy regression harness against
`scipy==1.15.3`, and builds PyInstaller binaries on Windows, macOS, and Linux.
Each build job uploads its artifact (`how-many-windows`, `how-many-macos`, and
`how-many-linux`) for download.

## Configuration

Runtime preferences are persisted to `~/.how_many_config.json`. The file stores
analysis parameters (stripe width, auto-analysis toggle, minimum length, and so
on) together with UI choices (manual count, line thickness, tick length, and
always-on-top state).
