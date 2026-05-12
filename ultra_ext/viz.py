"""Model-agnostic per-item image inspector with 3 viewing modes.

Usage pattern: the caller pre-builds one zero-arg ``callback`` per image
(closing over whatever model / preprocessing / rendering it wants).  The
inspector knows nothing about models, datasets, or task-specific logic — it
just calls each callback to get a BGR ndarray (or a path to an existing image)
and dispatches to the chosen viewing mode.

Modes:

* ``iterative``    — overwrite a single file (``out_path``); open it once via
  the ``code`` CLI so VS Code's image preview auto-reloads as you step with
  Enter/p/q/<num>.

* ``batch``        — wipe ``out_dir``, render every callback, print absolute
  paths.  Returns the list.

* ``batch_finder`` — like ``batch`` then open the folder in macOS Finder +
  Quick Look gallery (falls back to ``open <dir>`` elsewhere).

Both batch modes wipe ``out_dir`` first so stale images from a previous run
never leak into the new gallery.

Example::

    from ultra_ext.viz import ImageInspector
    callbacks = [(lambda p=p: render(model, p)) for p in img_paths]
    ImageInspector(callbacks, labels=img_paths).iterative()
    # or .batch() / .batch_finder() / .run("batch_finder")
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

DEFAULT_FILE = "runs/temp/predict.jpg"
DEFAULT_DIR  = "runs/temp/predict"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
_MODES = ("iterative", "batch", "batch_finder")


class ImageInspector:
    """Drive a list of zero-arg image-producing callbacks in three viewing modes.

    Attributes:
        callbacks: Sequence of ``() -> np.ndarray | str`` (BGR ndarray or path).
        labels:    Optional per-item display strings (default: ``"#<i>"``).
        out_path:  File overwritten by ``iterative`` mode.
        out_dir:   Folder wiped and populated by the two ``batch`` modes.
        name_fn:   ``(idx, label) -> filename`` for batch output naming.
    """

    def __init__(
        self,
        callbacks: Sequence[Callable[[], Any]],
        labels: Sequence[str] | None = None,
        out_path: str = DEFAULT_FILE,
        out_dir: str = DEFAULT_DIR,
        name_fn: Callable[[int, str], str] | None = None,
    ) -> None:
        self.callbacks = list(callbacks)
        self.labels = list(labels) if labels else [f"#{i}" for i in range(len(self.callbacks))]
        if len(self.labels) != len(self.callbacks):
            raise ValueError("labels and callbacks must have the same length")
        self.out_path = os.path.abspath(out_path)
        self.out_dir  = os.path.abspath(out_dir)
        self.name_fn  = name_fn or self._default_name

    # ── public modes ────────────────────────────────────────────────────────

    def iterative(self) -> None:
        """Overwrite ``out_path`` per item; open it once in VS Code.

        VS Code's image preview auto-reloads on file change, so we just write
        the file each step and let VS Code handle the rest.
        """
        if not self.callbacks:
            print("[ImageInspector.iterative] no items")
            return
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        print(f"[ImageInspector.iterative] writing → {self.out_path}  "
              f"({len(self.callbacks)} items)")

        # Seed the file so VS Code has something to open, then open it once.
        self._write(self.out_path, self.callbacks[0]())
        subprocess.run(["code", "-r", self.out_path], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        i, n = 0, len(self.callbacks)
        while 0 <= i < n:
            print(f"\n[{i + 1}/{n}] {self.labels[i]}")
            try:
                self._write(self.out_path, self.callbacks[i]())
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {e!r}")
            ans = input("[Enter]=next, p=prev, q=quit, <num>=jump: ").strip()
            if ans in ("q", "Q"):
                break
            if ans in ("p", "P"):
                i = max(i - 1, 0)
            elif ans.isdigit():
                i = max(0, min(int(ans) - 1, n - 1))
            else:
                i += 1

    def batch(self, print_paths: bool = True) -> list[str]:
        """Wipe ``out_dir``, render all callbacks, return abs paths."""
        self._wipe(self.out_dir)
        print(f"[ImageInspector.batch] → {self.out_dir}  "
              f"(wiped, {len(self.callbacks)} items)")
        saved: list[str] = []
        for idx, (cb, label) in enumerate(zip(self.callbacks, self.labels)):
            out_path = os.path.join(self.out_dir, self.name_fn(idx, label))
            try:
                self._write(out_path, cb())
                saved.append(out_path)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED on {label}: {e!r}")
        if print_paths:
            print(f"\n[ImageInspector.batch] saved {len(saved)} images:")
            for p in saved:
                print(p)
        return saved

    def batch_finder(self) -> list[str]:
        """Run ``batch`` then open the folder in Finder + Quick Look."""
        saved = self.batch(print_paths=False)
        self._open_finder_gallery(self.out_dir)
        return saved

    def run(self, mode: str = "iterative"):
        """Dispatch to one of the modes by name."""
        if mode not in _MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {_MODES}")
        return getattr(self, mode)()

    # ── private helpers (all stateless) ─────────────────────────────────────

    @staticmethod
    def _default_name(idx: int, label: str) -> str:
        """Numeric-prefixed sanitised label so order is preserved on disk."""
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
        if not safe.lower().endswith(tuple(_IMG_EXTS)):
            safe += ".jpg"
        return f"{idx:04d}_{safe}"

    @staticmethod
    def _write(out_path: str, result: Any) -> None:
        """Write `result` (BGR ndarray or src path) to `out_path`."""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if isinstance(result, np.ndarray):
            cv2.imwrite(out_path, result)
        elif isinstance(result, (str, Path)):
            shutil.copy(str(result), out_path)
        else:
            raise TypeError(
                f"callback must return ndarray or path, got {type(result).__name__}"
            )

    @staticmethod
    def _wipe(path: str) -> None:
        p = Path(path)
        if p.exists():
            for f in p.iterdir():
                (f.unlink() if f.is_file() else shutil.rmtree(f))
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _open_finder_gallery(dir_path: str) -> None:
        abs_dir = os.path.abspath(dir_path)
        if sys.platform != "darwin":
            print(f"(no Finder on {sys.platform}; opening folder)")
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", abs_dir],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        images = sorted(str(p) for p in Path(abs_dir).iterdir()
                        if p.suffix.lower() in _IMG_EXTS)
        subprocess.run(["open", abs_dir], check=False)
        if images:
            # qlmanage -p = macOS Quick Look gallery; arrow keys cycle images.
            subprocess.Popen(["qlmanage", "-p", *images],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
