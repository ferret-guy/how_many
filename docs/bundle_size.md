# Executable size analysis

The PyInstaller build script (`scripts/build_exe.py`) now prints a categorized
breakdown of the one-file bundle after each build. Run it with the project
managed virtual environment to regenerate the executable and inspect the size
composition:

```bash
uv run python scripts/build_exe.py
```

The command finishes by summarizing the compressed overlay, including the
largest contributors (Qt, NumPy, ICU, etc.) and any optional Qt plugins that
were pruned before packing.

Typical output on Linux shows the optimized bundle landing just under 60 MiB.
The heavy hitters are the Qt shared libraries and their ICU dependencies,
followed by NumPy's compiled extensions. The summary highlights these pieces so
it is easy to judge the trade-offs of deeper pruning or reimplementation work.
If a component still appears under "Potentially removable components," verify
its shared-library dependencies (for example with `ldd` on Linux or `otool -L`
on macOS) before removing it from the build: many Qt libraries are required by
others even if the application does not import their Python modules.
