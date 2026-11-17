#!/usr/bin/env python3
"""Reproduce the recovered Table XII boundary-region distribution."""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_paths import KITTI_2015


EXPECTED_SELECTED = [
    "000001_10", "000006_10", "000024_10", "000023_10", "000003_10",
    "000021_10", "000037_10", "000036_10", "000026_10", "000038_10",
]


@dataclass
class SampleStats:
    sample_id: str
    score: float
    suppression_ratio_nonshared: float
    shared_fraction: float
    flow_only_fraction: float
    disp_only_fraction: float
    nonboundary_fraction: float


def load_flow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(path)
    valid = data[..., 0] > 0
    flow = (data[..., 2:0:-1].astype(np.float32) - 32768.0) / 64.0
    flow[~valid] = 0.0
    return flow, valid


def load_disparity(path: Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(path)
    return data.astype(np.float32) / 256.0


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def normalize01(image: np.ndarray) -> np.ndarray:
    lo, hi = float(np.percentile(image, 1)), float(np.percentile(image, 99))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def analyze_sample(sample_id: str, root: Path) -> SampleStats:
    image = cv2.imread(str(root / "image_2" / f"{sample_id}.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(root / "image_2" / f"{sample_id}.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    flow, flow_valid = load_flow(root / "flow_occ" / f"{sample_id}.png")
    disparity = load_disparity(root / "disp_occ_0" / f"{sample_id}.png")

    raw_gradient = sobel_magnitude(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0)
    raw_normalized = normalize01(raw_gradient)
    flow_gradient = np.sqrt(sobel_magnitude(flow[..., 0]) ** 2 + sobel_magnitude(flow[..., 1]) ** 2)
    disparity_gradient = sobel_magnitude(disparity)
    disparity_valid = disparity > 0

    flow_boundary = (flow_gradient >= np.percentile(flow_gradient[flow_valid], 90.0)) & flow_valid
    disparity_boundary = (disparity_gradient >= np.percentile(disparity_gradient[disparity_valid], 90.0)) & disparity_valid
    flow_band, disparity_band = dilate(flow_boundary, 5), dilate(disparity_boundary, 5)
    shared = flow_band & disparity_band
    flow_only, disparity_only = flow_band & ~disparity_band, disparity_band & ~flow_band
    nonboundary = flow_valid & disparity_valid & ~(flow_band | disparity_band)

    shared_support = cv2.GaussianBlur(shared.astype(np.float32), (0, 0), sigmaX=5, sigmaY=5)
    nonshared_support = cv2.GaussianBlur((flow_only | disparity_only).astype(np.float32), (0, 0), sigmaX=5, sigmaY=5)
    modulated = raw_normalized * np.clip(0.12 + 0.68 * shared_support + 0.20 * raw_normalized - 0.55 * nonshared_support, 0.0, 1.0)

    def mean_in(mask: np.ndarray) -> float:
        return float(modulated[mask].mean()) if mask.any() else 0.0

    raw_shared = float(raw_normalized[shared].mean()) if shared.any() else 0.0
    raw_flow = float(raw_normalized[flow_only].mean()) if flow_only.any() else 0.0
    raw_disparity = float(raw_normalized[disparity_only].mean()) if disparity_only.any() else 0.0
    gated_flow, gated_disparity = mean_in(flow_only), mean_in(disparity_only)
    score = mean_in(shared) - 0.55 * (gated_flow + gated_disparity) / 2.0
    score += 0.25 * (raw_shared - (raw_flow + raw_disparity) / 2.0)
    return SampleStats(
        sample_id, score,
        float(1.0 - (gated_flow + gated_disparity) / (raw_flow + raw_disparity + 1e-6)),
        float(shared.mean()), float(flow_only.mean()), float(disparity_only.mean()), float(nonboundary.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=("XII", "XIII"), default="XII")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--data-root", type=Path, default=Path(KITTI_2015))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / f"boundary_analysis_{args.table}_{datetime.now():%Y%m%d_%H%M%S}"
    if args.table == "XIII":
        child = subprocess.run(
            [sys.executable, "-u", str(ROOT / "tables-XII-XIII" / "run_40sample_table.py"),
             "--table", "XIII", "--device", args.device],
            cwd=ROOT, stdout=subprocess.PIPE, text=True, env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        if child.returncode:
            if child.stdout:
                sys.stdout.write(child.stdout)
            raise SystemExit(child.returncode)
        report = child.stdout
        output.mkdir(parents=True, exist_ok=False)
        (output / "terminal_metrics.txt").write_text(report, encoding="utf-8")
        sys.stdout.write(report)
        return
    candidates = [path.stem for path in sorted((args.data_root / "flow_occ").glob("*.png"))[:40]]
    if len(candidates) != 40:
        raise SystemExit(f"Expected 40 KITTI flow files in {args.data_root / 'flow_occ'}, found {len(candidates)}.")
    rows = [analyze_sample(sample_id, args.data_root) for sample_id in candidates]
    selected = sorted(rows, key=lambda row: (row.score, row.suppression_ratio_nonshared), reverse=True)[:10]
    if [row.sample_id for row in selected] != EXPECTED_SELECTED:
        raise SystemExit("The recovered original Table XII selected-sample order was not reproduced.")

    columns = ("shared_fraction", "flow_only_fraction", "disp_only_fraction", "nonboundary_fraction")
    displayed = {column: round(float(np.mean([getattr(row, column) for row in selected])) * 100.0, 2) for column in columns}
    boundary_total = sum(displayed[column] for column in columns[:3])
    shares = {column: displayed[column] * 100.0 / boundary_total for column in columns[:3]}
    report = "\n".join([
        "CALCULATED RESULTS", "=========================================",
        f"shared_boundary_valid_pixels : {displayed['shared_fraction']:.2f}%",
        f"shared_boundary_band_share : {shares['shared_fraction']:.2f}%",
        f"flow_only_boundary_valid_pixels : {displayed['flow_only_fraction']:.2f}%",
        f"flow_only_boundary_band_share : {shares['flow_only_fraction']:.2f}%",
        f"disparity_only_boundary_valid_pixels : {displayed['disp_only_fraction']:.2f}%",
        f"disparity_only_boundary_band_share : {shares['disp_only_fraction']:.2f}%",
        f"nonboundary_control_valid_pixels : {displayed['nonboundary_fraction']:.2f}%",
    ]) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "terminal_metrics.txt").write_text(report, encoding="utf-8")
    sys.stdout.write(report)


if __name__ == "__main__":
    main()
