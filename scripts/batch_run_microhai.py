#!/usr/bin/env python3
"""Batch-run run_microhai.py on every MCP2 / MCP3 volume in joint_nii_file.

Put this file next to run_microhai.py. Input volumes live in ./joint_nii_file.
Each volume gets its own result folder under ./joint_nii_results/<stem>/.

    python batch_run_microhai.py
    python batch_run_microhai.py --input joint_nii_file --output joint_nii_results
    python batch_run_microhai.py --skip-existing

MCP_4 and any name that is not MCP2/MCP3 are skipped. A failed case is logged
and the batch continues. A summary is written to joint_nii_results/batch_log.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_microhai import mcp_index, run, stem_of


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    if h:
        return "{}:{:02d}:{:05.2f}".format(h, m, s)
    return "{:02d}:{:05.2f}".format(m, s)


def is_nifti(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def collect(input_dir: str) -> tuple[list[str], list[str]]:
    selected, skipped = [], []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path) or not is_nifti(name):
            continue
        try:
            mcp_index(stem_of(path))
        except ValueError:
            skipped.append(path)
            continue
        selected.append(path)
    return selected, skipped


def main() -> int:
    p = argparse.ArgumentParser(description="Run micro-HAI on every MCP2/MCP3 NIfTI in a folder.")
    p.add_argument("--input", default=os.path.join(HERE, "joint_nii_file"), help="Folder of source .nii files")
    p.add_argument("--output", default=os.path.join(HERE, "joint_nii_results"), help="Root for one folder per volume")
    p.add_argument("--template-dir", default=HERE, help="Directory with mean_mcp2.ply and mean_mcp3.ply")
    p.add_argument("--percentile", type=float, default=75.0)
    p.add_argument("--min-points", type=int, default=20)
    p.add_argument("--point-radius", type=float, default=1.0)
    p.add_argument("--skip-existing", action="store_true", help="Skip a volume if its label NIfTI already exists")
    args = p.parse_args()

    input_dir = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)
    if not os.path.isdir(input_dir):
        print("Input folder not found: {}".format(input_dir), file=sys.stderr)
        return 2
    os.makedirs(output_root, exist_ok=True)

    selected, skipped = collect(input_dir)
    print("MCP2/MCP3 volumes: {}".format(len(selected)))
    print("Skipped (not MCP2/MCP3): {}".format(len(skipped)))
    for path in skipped:
        print("  skip {}".format(os.path.basename(path)))

    log_path = os.path.join(output_root, "batch_log.csv")
    rows = []
    failed = 0
    batch_started = time.perf_counter()
    for i, nii_path in enumerate(selected, 1):
        stem = stem_of(nii_path)
        which = mcp_index(stem)
        case_dir = os.path.join(output_root, stem)
        label_path = os.path.join(case_dir, stem + "_microhai_labels.nii.gz")
        print("\n[{}/{}] MCP{}  {}".format(i, len(selected), which, os.path.basename(nii_path)))
        if args.skip_existing and os.path.isfile(label_path):
            print("  exists, skip")
            rows.append((os.path.basename(nii_path), which, "skipped", "0", case_dir, ""))
            continue
        os.makedirs(case_dir, exist_ok=True)
        started = time.perf_counter()
        try:
            written = run(
                nii_path,
                template_dir=args.template_dir,
                out_dir=case_dir,
                percentile=args.percentile,
                min_points=args.min_points,
                point_radius=args.point_radius,
                visualize=False,
            )
            elapsed = time.perf_counter() - started
            print("  time {}".format(fmt_seconds(elapsed)))
            rows.append((os.path.basename(nii_path), which, "ok", "{:.2f}".format(elapsed), written, ""))
        except Exception as exc:
            failed += 1
            elapsed = time.perf_counter() - started
            print("  FAILED after {}: {}".format(fmt_seconds(elapsed), exc))
            traceback.print_exc()
            rows.append((os.path.basename(nii_path), which, "failed", "{:.2f}".format(elapsed), case_dir, str(exc)))

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nii", "mcp", "status", "seconds", "output", "error"])
        writer.writerows(rows)
    timed = [float(r[3]) for r in rows if r[2] in ("ok", "failed")]
    total = time.perf_counter() - batch_started
    print("\nDone. ok={} failed={} skipped_existing={}".format(
        sum(1 for r in rows if r[2] == "ok"),
        failed,
        sum(1 for r in rows if r[2] == "skipped"),
    ))
    print("Total time: {}".format(fmt_seconds(total)))
    if timed:
        print("Mean per case: {}  ({:.1f} s)".format(fmt_seconds(sum(timed) / len(timed)), sum(timed) / len(timed)))
        print("Slowest: {} s  fastest: {} s".format(max(timed), min(timed)))
    print("Log: {}".format(log_path))
    print("Finished {}".format(datetime.now().isoformat(timespec="seconds")))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
