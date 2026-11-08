"""Strip the presentation-style comments (【讲解要点】banners, ★ markers) from the code.

The technical content stays, but written as ordinary developer comments:
  · the banner blocks  (# ═══════ ... ═══════)  are removed
  · "★ 讲解点：" / "★ " prefixes are removed, "讲的时候…/答辩…" wording is dropped

Usage::

    python scripts/strip_presentation_comments.py          # dry run, prints what changes
    python scripts/strip_presentation_comments.py --write   # apply
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

FILES = [
    "core/flows/paths.py",
    "core/flows/consistency.py",
    "core/flows/samplers.py",
    "core/flows/policy.py",
    "core/flow_dynaguide.py",
    "core/flow_energies.py",
    "core/satellite/env.py",
    "core/satellite/expert.py",
    "core/satellite/latent.py",
    "scripts/satellite/run_flow_dynaguide.py",
    "robomimic/robomimic/algo/flow_policy.py",
]

BANNER = re.compile(r"^# ═{10,}")

PHRASE_FIXES = [
    ("讲的时候可以强调：", ""),
    ("讲的时候按三块走：", "结构分三块："),
    ("讲的时候按这个顺序：", ""),
    ("讲的时候一定要说清：", ""),
    ("讲的时候可以顺带说清边界：", "边界说明："),
    ("答辩时可强调：", ""),
    ("答辩常被问“为什么怕大梯度”，答案就在这里。", ""),
    ("★ 讲解要点（", "("),
    ("★ 讲解点：", ""),
    ("★ 讲解点", ""),
    ("★ ", ""),
]


def strip_banner(text: str) -> tuple[str, int]:
    lines = text.split("\n")
    out, i, removed = [], 0, 0
    while i < len(lines):
        if BANNER.match(lines[i]):
            j = i + 1
            while j < len(lines) and not BANNER.match(lines[j]):
                j += 1
            if j < len(lines):
                removed += 1
                i = j + 1
                # drop a single blank line right after the banner
                if i < len(lines) and lines[i].strip() == "":
                    i += 1
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--root", default="C:/代码修改/dynaguide")
    args = ap.parse_args()
    root = Path(args.root)

    for rel in FILES:
        path = root / rel
        original = path.read_text(encoding="utf-8")
        text, banners = strip_banner(original)
        for old, new in PHRASE_FIXES:
            text = text.replace(old, new)
        text = re.sub(r"[ \t]+\n", "\n", text)  # trailing spaces
        changed = text != original
        marks = sum(original.count(k) for k in ("★", "讲解", "答辩", "讲的时候"))
        print(f"{rel:52s} banner={banners} presentation_marks={marks} changed={changed}")
        if args.write and changed:
            path.write_text(text, encoding="utf-8")
    if not args.write:
        print("\n(dry run — 加 --write 才会写入)")


if __name__ == "__main__":
    main()
