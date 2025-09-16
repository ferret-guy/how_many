# How Many

- Always-on-top overlay for counting evenly spaced structures directly on any image.
- Drag a line across the features to let the tool analyse the stripe underneath and suggest likely counts.

## Demo

![How Many demo](docs/images/how-many-demo.gif)

- run with `uvx --from git+https://github.com/ferret-guy/how_many how-many`
- Or grab a pre-built package from a [release](https://github.com/ferret-guy/how_many/releases).

## Feature highlights

- Transparent overlay window with draggable endpoints, configurable stripe, and tick markers.
- Control dialog bundling analysis controls, profile visualisation, and a quick-reference help tab.
- Automatic FFT and autocorrelation suggestions (works 50% of the time, every time).
- Manual item-count override with instant overlay feedback.
- Profile tab exposing the 1-D signal and the candidate list powering the UI.
- Keyboard shortcuts for analysis, stripe adjustments, and manual counts.
- Preferences saved in `~/.how_many_config.json` so settings persist between runs.

## UI at a glance

- **Overlay window**
  - Drag endpoints or the whole line; hold `Ctrl` to snap to 0°/45°/90°/135°.
  - Tick markers always include endpoints, and the HUD shows `L`, `W`, and `N` offset from the stripe.
- **Control dialog**
  - Controls tab: candidate dropdown, manual count, analyse-now button, auto-analysis toggle, and stripe/style options.
  - Profile tab: 1-D intensity graph with vertical markers and the ranked candidate list.
  - Help tab: compact usage notes and shortcut reminders.
- **Shortcuts**
  - `A` — Trigger an analysis.
  - `+` / `-` — Increase or decrease the item count `N`.
  - `W` / `Q` — Widen or narrow the stripe `W`.
  - `Esc` — Close the overlay.

## Running

- **Setup (uv):** `uv sync --group build --group test`
- **Run (uv):** `uv run python -m how_many`
- **Or build an executable**: `uv run python scripts/build_exe.py`

## Continuous integration

- GitHub Actions (see `.github/workflows/build.yml`) uses [uv](https://github.com/astral-sh/uv) to run the SciPy regression harness against `scipy==1.15.3`.
- Windows, macOS, and Linux jobs build PyInstaller executables and upload them as downloadable artifacts.
