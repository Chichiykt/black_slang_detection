#!/usr/bin/env python3
"""Run hyper-ball clustering on normalized feature tensors.

The source algorithm in ``粒球.md`` assumes that each internal row has an id
column before the feature vector.  The public input here is just ``[n, m]``;
this script adds that id column internally and exports only feature-space
centers/radii/labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LABEL_NUM = 2
EPS = 1e-12


@dataclass
class BallNode:
    indices: np.ndarray
    children: tuple["BallNode", "BallNode"] | None = None


def read_simple_torch_tensor(path: Path) -> np.ndarray:
    """Read simple PyTorch zip tensors without importing torch.

    This is intentionally narrow: it supports the FloatStorage/LongStorage
    tensors used by the provided ``.pt`` files.
    """
    with zipfile.ZipFile(path) as archive:
        pkl_name = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        raw_name = next(name for name in archive.namelist() if name.endswith("data/0"))
        pkl = archive.read(pkl_name)
        raw = archive.read(raw_name)

    if b"FloatStorage" in pkl:
        dtype = np.float32
    elif b"LongStorage" in pkl:
        dtype = np.int64
    else:
        raise ValueError(f"Unsupported tensor storage in {path}")

    shape = parse_torch_shape_from_pickle(pkl)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def parse_torch_shape_from_pickle(pkl: bytes) -> tuple[int, ...]:
    """Extract the tensor shape from the small protocol-2 pickle payload."""
    import pickletools

    values_after_offset: list[int] = []
    found_storage_ref = False
    skipped_storage_offset = False

    for op, arg, _pos in pickletools.genops(pkl):
        if op.name == "BINPERSID":
            found_storage_ref = True
            continue
        if not found_storage_ref:
            continue
        if op.name.startswith("BININT") or op.name == "LONG1":
            if not skipped_storage_offset:
                skipped_storage_offset = True
                continue
            values_after_offset.append(int(arg))
            continue
        if not skipped_storage_offset:
            continue
        if op.name == "TUPLE1":
            return (values_after_offset[-1],)
        if op.name == "TUPLE2":
            return tuple(values_after_offset[-2:])
        if op.name == "TUPLE3":
            return tuple(values_after_offset[-3:])
        if op.name == "TUPLE" and values_after_offset:
            return tuple(values_after_offset)
        if op.name == "EMPTY_TUPLE":
            break

    raise ValueError("Could not parse tensor shape from torch pickle")


def normalize_center(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    norm = np.linalg.norm(center)
    if norm <= EPS:
        return points[0].copy()
    return center / norm


def angular_distances(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    dots = np.clip(points @ center, -1.0, 1.0)
    return np.arccos(dots)


def center_radius(features: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, float, float]:
    points = features[indices]
    center = normalize_center(points)
    distances = angular_distances(points, center)
    mean_radius = float(distances.mean()) if len(distances) else 0.0
    max_radius = float(distances.max()) if len(distances) else 0.0
    dm = mean_radius if mean_radius != 0.0 else max_radius
    return center.astype(np.float32), mean_radius, dm


def split_ball(features: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = features[indices]
    n = len(indices)
    if n < 2:
        return indices.copy(), np.array([], dtype=np.int64)

    similarities = points @ points.T
    np.fill_diagonal(similarities, np.inf)
    first_local, second_local = np.unravel_index(np.argmin(similarities), similarities.shape)

    first_scores = points @ points[first_local]
    second_scores = points @ points[second_local]
    first_mask = first_scores > second_scores

    first_child = indices[first_mask]
    second_child = indices[~first_mask]
    if len(first_child) == 0 or len(second_child) == 0:
        return first_child.astype(np.int64), second_child.astype(np.int64)

    first_center, first_radius, _ = center_radius(features, first_child)
    second_center, second_radius, _ = center_radius(features, second_child)
    first_distances = angular_distances(points, first_center)
    second_distances = angular_distances(points, second_center)
    overlap_mask = (first_distances <= first_radius) & (second_distances <= second_radius)

    if np.any(overlap_mask):
        overlap_indices = indices[overlap_mask]
        first_overlap = overlap_indices[first_mask[overlap_mask]]
        second_overlap = overlap_indices[~first_mask[overlap_mask]]
        first_child = np.concatenate([first_child, second_overlap])
        second_child = np.concatenate([second_child, first_overlap])

    return first_child.astype(np.int64), second_child.astype(np.int64)


def build_tree(features: np.ndarray, indices: np.ndarray, min_synonyms: int) -> BallNode:
    node = BallNode(indices=indices.astype(np.int64))
    if len(indices) < min_synonyms * 2:
        return node

    first_child, second_child = split_ball(features, indices)
    if len(first_child) == 0 or len(second_child) == 0:
        return node

    _, _, parent_dm = center_radius(features, indices)
    _, _, first_dm = center_radius(features, first_child)
    _, _, second_dm = center_radius(features, second_child)
    child_weighted_dm = (len(first_child) * first_dm + len(second_child) * second_dm) / (
        len(first_child) + len(second_child)
    )

    if child_weighted_dm < parent_dm:
        node.children = (
            build_tree(features, first_child, min_synonyms),
            build_tree(features, second_child, min_synonyms),
        )
    return node


def collect_leaves(node: BallNode, min_synonyms: int) -> list[np.ndarray]:
    if node.children is None or len(node.indices) < min_synonyms * 2:
        return [node.indices]
    leaves: list[np.ndarray] = []
    leaves.extend(collect_leaves(node.children[0], min_synonyms))
    leaves.extend(collect_leaves(node.children[1], min_synonyms))
    return leaves


def ball_labels(labels: np.ndarray, leaves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    soft_labels = np.zeros((len(leaves), LABEL_NUM), dtype=np.float32)
    hard_labels = np.zeros(len(leaves), dtype=np.int64)
    for i, indices in enumerate(leaves):
        counts = np.bincount(labels[indices], minlength=LABEL_NUM).astype(np.float64)
        soft = counts / counts.sum()
        soft_labels[i] = soft
        hard_labels[i] = int(np.argmax(counts))
    return soft_labels, hard_labels


def summarize_balls(features: np.ndarray, leaves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros((len(leaves), features.shape[1]), dtype=np.float32)
    radii = np.zeros(len(leaves), dtype=np.float32)
    for i, indices in enumerate(leaves):
        center, radius, _ = center_radius(features, indices)
        centers[i] = center
        radii[i] = radius
    return centers, radii


def assign_labels(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    hard_labels: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    predicted = np.zeros(len(features), dtype=np.int64)
    assigned_balls = np.zeros(len(features), dtype=np.int64)
    outside_radius = 0
    multi_ball = 0

    for start in range(0, len(features), block_size):
        end = min(start + block_size, len(features))
        block = features[start:end]
        similarities = np.clip(block @ centers.T, -1.0, 1.0)
        distances = np.arccos(similarities)
        in_radius = distances <= radii[None, :]
        candidate_counts = in_radius.sum(axis=1)

        nearest_any = distances.argmin(axis=1)
        masked_distances = np.where(in_radius, distances, np.inf)
        nearest_candidate = masked_distances.argmin(axis=1)

        no_candidate = candidate_counts == 0
        nearest_candidate[no_candidate] = nearest_any[no_candidate]

        outside_radius += int(no_candidate.sum())
        multi_ball += int((candidate_counts > 1).sum())
        assigned_balls[start:end] = nearest_candidate
        predicted[start:end] = hard_labels[nearest_candidate]

    accuracy = float((predicted == labels).mean())
    metrics = {
        "accuracy": accuracy,
        "sample_count": int(len(labels)),
        "ball_count": int(len(centers)),
        "outside_radius_count": int(outside_radius),
        "multi_ball_count": int(multi_ball),
    }
    return predicted, metrics


def save_run(
    output_dir: Path,
    centers: np.ndarray,
    radii: np.ndarray,
    soft_labels: np.ndarray,
    hard_labels: np.ndarray,
    predicted_labels: np.ndarray,
    metrics: dict[str, int | float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "centers.npy", centers)
    np.save(output_dir / "radii.npy", radii)
    np.save(output_dir / "soft_labels.npy", soft_labels)
    np.save(output_dir / "hard_labels.npy", hard_labels)
    np.save(output_dir / "predicted_labels.npy", predicted_labels)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyper-ball clustering and evaluate labels.")
    parser.add_argument("--feature-pt", type=Path, default=Path("hidden_second_last_normalized.pt"))
    parser.add_argument("--label-pt", type=Path, default=Path("true_labels.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--min-synonyms", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--block-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = read_simple_torch_tensor(args.feature_pt).astype(np.float32, copy=False)
    labels = read_simple_torch_tensor(args.label_pt).astype(np.int64, copy=False)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")
    if labels.shape != (features.shape[0],):
        raise ValueError(f"Label shape {labels.shape} does not match features {features.shape}")

    min_values = sorted(set(args.min_synonyms))
    min_for_tree = min(min_values)
    root = build_tree(features, np.arange(len(features), dtype=np.int64), min_for_tree)

    summaries: list[dict[str, int | float]] = []
    best_dir: Path | None = None
    best_accuracy = -1.0

    for min_synonyms in min_values:
        leaves = collect_leaves(root, min_synonyms)
        centers, radii = summarize_balls(features, leaves)
        soft_labels, hard_labels = ball_labels(labels, leaves)
        predicted_labels, metrics = assign_labels(
            features, labels, centers, radii, hard_labels, args.block_size
        )
        metrics = {"min_synonyms": int(min_synonyms), **metrics}
        run_dir = args.output_dir / f"min_synonyms_{min_synonyms}"
        save_run(run_dir, centers, radii, soft_labels, hard_labels, predicted_labels, metrics)
        summaries.append(metrics)

        if float(metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(metrics["accuracy"])
            best_dir = run_dir

        print(
            "min_synonyms={min_synonyms} balls={ball_count} "
            "accuracy={accuracy:.6f} outside_radius={outside_radius_count} "
            "multi_ball={multi_ball_count}".format(**metrics),
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if best_dir is not None:
        for name in [
            "centers.npy",
            "radii.npy",
            "soft_labels.npy",
            "hard_labels.npy",
            "predicted_labels.npy",
            "metrics.json",
        ]:
            shutil.copy2(best_dir / name, args.output_dir / name)
            shutil.copy2(best_dir / name, Path(name))


if __name__ == "__main__":
    main()
