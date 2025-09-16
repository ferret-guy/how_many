"""Utility script for building the how_many executable via PyInstaller."""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import Sequence


def _version_tuple(version: str) -> tuple[int, int, int, int]:
    parts: list[int] = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 4:
            break
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts[:4])


def _write_version_file(version: str) -> Path:
    numbers = _version_tuple(version)
    contents = dedent(
        f"""
        VSVersionInfo(
          ffi=FixedFileInfo(
            filevers={numbers},
            prodvers={numbers},
            mask=0x3F,
            flags=0x0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0)
          ),
          kids=[
            StringFileInfo(
              [
                StringTable(
                  '040904B0',
                  [
                    StringStruct('CompanyName', 'how_many'),
                    StringStruct('FileDescription', 'how_many overlay'),
                    StringStruct('FileVersion', '{version}'),
                    StringStruct('InternalName', 'how_many'),
                    StringStruct('OriginalFilename', 'how_many.exe'),
                    StringStruct('ProductName', 'how_many'),
                    StringStruct('ProductVersion', '{version}')
                  ]
                )
              ]
            ),
            VarFileInfo([VarStruct('Translation', [1033, 1200])])
          ]
        )
        """
    ).strip()

    with NamedTemporaryFile("w", delete=False, suffix=".txt") as fp:
        fp.write(contents)
        return Path(fp.name)


@dataclass
class BuildConfig:
    name: str = "how_many"
    entry: Path = Path("src/how_many/app.py")


@dataclass
class ArchiveEntry:
    name: str
    length: int
    uncompressed_length: int
    typecode: str
    is_compressed: bool


def _format_size(num: int) -> str:
    if num >= 1024**2:
        return f"{num / (1024 ** 2):.2f} MiB"
    if num >= 1024:
        return f"{num / 1024:.1f} KiB"
    return f"{num} B"


def _category_for_entry(name: str) -> str:
    if name.startswith(("pyimod", "pyi_rth")) or name in {"struct", "pyz"}:
        return "PyInstaller bootstrap"
    if name == "PYZ.pyz":
        return "Python modules (PYZ)"
    if name.startswith("PySide6/") or name.startswith("shiboken6") or name.startswith("libQt6"):
        return "PySide6"
    if name.startswith("numpy") or name.startswith("libgfortran") or name.startswith("libquadmath"):
        return "NumPy"
    if name.startswith(("libpython", "python3")):
        return "CPython runtime"
    if name.startswith(("libssl", "libcrypto")):
        return "OpenSSL"
    if name == "base_library.zip":
        return "Python stdlib"
    if name == "app" or name.startswith("how_many"):
        return "Application code"
    if name.startswith("libX"):
        return "X11 libraries"
    if name.startswith("lib"):
        return "System libraries"
    prefix = name.split("/", 1)[0]
    return prefix or name


def _aggregate_by_prefix(
    entries: Sequence[ArchiveEntry], *, depth: int
) -> list[tuple[str, int]]:
    totals: dict[str, int] = defaultdict(int)
    for entry in entries:
        parts = entry.name.split("/")
        if len(parts) >= depth:
            key = "/".join(parts[:depth])
        else:
            key = entry.name
        totals[key] += entry.length
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def _aggregate_by_category(entries: Sequence[ArchiveEntry]) -> list[tuple[str, int]]:
    totals: dict[str, int] = defaultdict(int)
    for entry in entries:
        key = _category_for_entry(entry.name)
        totals[key] += entry.length
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


OPTIONAL_PREFIXES = (
    "PySide6/Qt/plugins/generic/",
    "PySide6/Qt/plugins/iconengines/",
    "PySide6/Qt/plugins/imageformats/",
    "PySide6/Qt/plugins/wayland-",
)

OPTIONAL_TOKENS = (
    "libqwayland",
    "libqeglfs",
    "libqvnc",
    "libqminimal",
    "libqoffscreen",
    "libqvkkhrdisplay",
    "libqxcb-egl-integration",
    "libqxcb-glx-integration",
    "Qt6DBus",
    "Qt6Network",
)


def _collect_optional_components(entries: Sequence[ArchiveEntry]) -> list[tuple[str, int]]:
    totals: dict[str, int] = defaultdict(int)
    for entry in entries:
        name = entry.name
        for prefix in OPTIONAL_PREFIXES:
            if name.startswith(prefix):
                totals[prefix] += entry.length
                break
        else:
            for token in OPTIONAL_TOKENS:
                if token in name:
                    totals[token] += entry.length
                    break
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def _print_component_breakdown(
    title: str,
    entries: Sequence[ArchiveEntry],
    *,
    total_size: int,
    depth: int,
    max_items: int = 6,
) -> None:
    if not entries:
        return
    subtotal = sum(entry.length for entry in entries)
    if subtotal == 0:
        return
    print(f"\n  {title}: {_format_size(subtotal)} ({subtotal / total_size:.1%} of overlay)")
    for prefix, size in _aggregate_by_prefix(entries, depth=depth)[:max_items]:
        print(
            f"    • {prefix}: {_format_size(size)} ({size / subtotal:.1%} of component)"
        )
    top_files = sorted(entries, key=lambda entry: entry.length, reverse=True)[:max_items]
    print("    Largest files:")
    for item in top_files:
        print(
            f"      - {item.name}: {_format_size(item.length)} ({item.length / total_size:.1%} of overlay)"
        )


def _parse_archive_listing(output: str) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue
        parts = [part.strip() for part in stripped.split(", ")]
        if len(parts) != 6:
            continue
        try:
            _, length, uncompressed, is_compressed, typecode, name = parts
            entry = ArchiveEntry(
                name=name.strip("'"),
                length=int(length),
                uncompressed_length=int(uncompressed),
                is_compressed=bool(int(is_compressed)),
                typecode=typecode.strip("'"),
            )
        except ValueError:
            continue
        entries.append(entry)
    return entries


def _inspect_bundle(bundle: Path) -> list[ArchiveEntry]:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller.utils.cliutils.archive_viewer",
        str(bundle),
        "-r",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "archive viewer failed")
    return _parse_archive_listing(result.stdout)


def _locate_bundle(dist_dir: Path, name: str) -> Path | None:
    candidates = [
        dist_dir / name,
        dist_dir / f"{name}.exe",
        dist_dir / f"{name}.app" / name,
        dist_dir / f"{name}.app" / "Contents" / "MacOS" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    for path in sorted(dist_dir.glob(f"{name}*"), key=lambda p: p.stat().st_size if p.is_file() else 0, reverse=True):
        if path.is_file():
            return path
    return None


def _summarize_bundle(entries: Sequence[ArchiveEntry]) -> None:
    if not entries:
        print("\nNo embedded files discovered; skipping bundle analysis.")
        return
    total_size = sum(entry.length for entry in entries)
    print("\nBundle contents (compressed overlay):")
    print(f"  Total size: {_format_size(total_size)} across {len(entries)} files")

    print("  Top-level composition:")
    for category, size in _aggregate_by_category(entries)[:10]:
        print(f"    - {category}: {_format_size(size)} ({size / total_size:.1%})")

    pyside_entries = [
        entry
        for entry in entries
        if entry.name.startswith("PySide6/")
        or entry.name.startswith("libQt6")
        or entry.name.startswith("shiboken6")
    ]
    numpy_entries = [entry for entry in entries if entry.name.startswith("numpy")]

    _print_component_breakdown(
        "PySide6 bundle",
        pyside_entries,
        total_size=total_size,
        depth=4,
        max_items=8,
    )
    _print_component_breakdown(
        "NumPy extension modules",
        numpy_entries,
        total_size=total_size,
        depth=2,
        max_items=6,
    )

    optional = _collect_optional_components(entries)
    if optional:
        print("\n  Potentially removable components (inspect before dropping):")
        for label, size in optional:
            print(f"    - {label}: {_format_size(size)} ({size / total_size:.1%})")


def main() -> None:
    cfg = BuildConfig()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    try:
        pkg = import_module("how_many")
        version = getattr(pkg, "__version__", "0.0.0")
    except Exception:  # pragma: no cover - build script safety net
        version = "0.0.0"

    version_file: Path | None = None
    script_path = root / cfg.entry

    qt_excludes = [
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtGraphs",
        "PySide6.QtLocation",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtNetwork",
        "PySide6.QtNfc",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtPositioning",
        "PySide6.QtPrintSupport",
        "PySide6.QtQml",
        "PySide6.QtQmlModels",
        "PySide6.QtQmlWorkerScript",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtQuickControls2",
        "PySide6.QtQuickWidgets",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtSql",
        "PySide6.QtStateMachine",
        "PySide6.QtSvg",
        "PySide6.QtTest",
        "PySide6.QtTextToSpeech",
        "PySide6.QtUiTools",
        "PySide6.QtVirtualKeyboard",
        "PySide6.QtWebChannel",
        "PySide6.QtWebEngine",
        "PySide6.QtWebSockets",
        "PySide6.QtXml",
        "PySide6.QtXmlPatterns",
    ]

    drop_binary_tokens = [
        "Qt6Pdf",
        "Qt6Qml",
        "Qt6Quick",
        "Qt6Svg",
        "Qt6VirtualKeyboard",
        "Qt6Wayland",
        "Qt6WlShell",
        "Qt6Egl",
        "Qt6EglFs",
        "libqwayland",
        "libqeglfs",
        "libqvnc",
        "libqminimal",
        "libqminimalegl",
        "libqoffscreen",
        "libqvkkhrdisplay",
        "libqxcb-egl-integration",
        "libqxcb-glx-integration",
        "libqlinuxfb",
        "libqgtk",
        "libqtuiotouchplugin",
        "libqevdev",
        "libqxdgdesktopportal",
    ]
    drop_binary_prefixes = [
        "PySide6/Qt/plugins/wayland-",
        "PySide6/Qt/plugins/egldeviceintegrations/",
        "PySide6/Qt/plugins/networkinformation/",
        "PySide6/Qt/plugins/platforminputcontexts/libqtvirtualkeyboardplugin",
        "PySide6/Qt/plugins/platformthemes/",
        "PySide6/Qt/plugins/iconengines/",
        "PySide6/Qt/plugins/imageformats/",
        "PySide6/Qt/plugins/generic/",
    ]
    drop_data_prefixes = [
        "PySide6/Qt/translations",
    ]

    spec_source = dedent(
        f"""
        # -*- mode: python ; coding: utf-8 -*-

        block_cipher = None


        def _filter(items, prefixes, tokens):
            kept = []
            for entry in items:
                name = entry[0]
                if any(name.startswith(prefix) for prefix in prefixes):
                    continue
                if any(token in name for token in tokens):
                    continue
                kept.append(entry)
            return items.__class__(kept)


        a = Analysis(
            ['{script_path}'],
            pathex=['{root / "src"}'],
            binaries=[],
            datas=[],
            hiddenimports=[],
            hookspath=[],
            hooksconfig={{}},
            runtime_hooks=[],
            excludes={qt_excludes!r},
            noarchive=False,
            optimize=0,
        )

        a.binaries = _filter(a.binaries, {drop_binary_prefixes!r}, {drop_binary_tokens!r})
        a.datas = _filter(a.datas, {drop_data_prefixes!r}, [])

        pyz = PYZ(a.pure)

        exe = EXE(
            pyz,
            a.scripts,
            a.binaries,
            a.datas,
            [],
            name='{cfg.name}',
            debug=False,
            bootloader_ignore_signals=False,
            strip=False,
            upx=True,
            upx_exclude=[],
            runtime_tmpdir=None,
            console=False,
            disable_windowed_traceback=False,
            argv_emulation=False,
            target_arch=None,
            codesign_identity=None,
            entitlements_file=None,
            version={{VERSION_FILE}},
        )
        """
    )

    spec_file: Path | None = None
    try:
        version_file = _write_version_file(version)
    except Exception:
        version_file = None

    version_literal = repr(str(version_file)) if version_file is not None else "None"
    spec_text = spec_source.replace("{VERSION_FILE}", version_literal)

    build_return = 1
    try:
        with NamedTemporaryFile("w", delete=False, suffix=".spec") as fp:
            fp.write(spec_text)
            spec_file = Path(fp.name)

        cmd = ["pyinstaller", "--clean", str(spec_file)]
        result = subprocess.run(cmd, check=False)
        build_return = result.returncode
    finally:
        if spec_file is not None and spec_file.exists():
            try:
                spec_file.unlink()
            except OSError:
                pass
        if version_file is not None and version_file.exists():
            try:
                version_file.unlink()
            except OSError:
                pass

    if build_return != 0:
        sys.exit(build_return)

    dist_dir = root / "dist"
    bundle = _locate_bundle(dist_dir, cfg.name)
    if bundle is None:
        print(f"Unable to locate the built executable in {dist_dir}", file=sys.stderr)
        return

    try:
        entries = _inspect_bundle(bundle)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"Bundle analysis failed: {exc}", file=sys.stderr)
        return

    print(f"\nAnalyzing bundled overlay at {bundle}")
    _summarize_bundle(entries)


if __name__ == "__main__":
    main()
