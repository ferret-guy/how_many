# How Many

- Always-on-top overlay for counting evenly spaced structures directly on any image.
- Drag a line across the features to let the tool analyse the stripe underneath and suggest likely counts.
- Example overlay:
  - ![How Many demo](https://storage.googleapis.com/generative-static-how_many/A8NwcEbY6nQf2vK1rNaREgXATuj2/uWrKX0HmUhQFPWaZYWn5ZNKI2OR2/3X2tFFyz3y4uwTszqfyk1E0x7s83/004Zwc3L6vgWzY5FSgjUHTrOCi12)

## Feature highlights

- Transparent overlay window with draggable endpoints, configurable stripe, and tick markers.
- Control dialog bundling analysis controls, profile visualisation, and a quick-reference help tab.
- Automatic FFT and autocorrelation suggestions with the off-by-one correction (items = cycles + 1).
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

## Usage

- `uv sync --group build --group test`
- `uv run python -m how_many`
- `uv run python scripts/build_exe.py`

## Continuous integration

- GitHub Actions (see `.github/workflows/build.yml`) uses [uv](https://github.com/astral-sh/uv) to run the SciPy regression harness against `scipy==1.15.3`.
- Windows, macOS, and Linux jobs build PyInstaller executables and upload them as downloadable artifacts.
