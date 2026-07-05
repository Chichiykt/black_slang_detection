#!/usr/bin/env python3
"""粒球分类器 — 基于预计算粒球（Hyper-Ball）对样本向量进行快速分类。

使用方式
--------
    from granular_ball_classifier import GranularBallClassifier

    clf = GranularBallClassifier("粒球聚类结果")          # 从目录加载
    label = clf.predict(sample_vector)                     # 单个样本 → int
    labels = clf.predict_batch(sample_matrix)              # 批量样本 → np.ndarray
    probs  = clf.predict_proba(sample_vector)              # 软标签概率

算法概要
--------
粒球分类器在角度空间（余弦 → arccos 距离）中预先将特征空间划分为若干
"超球"。预测时：
  1. 计算样本到每个球心的角度距离
  2. 找出包含该样本的所有球（距离 ≤ 半径）
  3. 若有候选球，取最近者；否则取全局最近球
  4. 返回该球的硬标签
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class PredictionDetail:
    """单次预测的详细信息。"""

    label: int
    """预测的分类标签。"""
    confidence: float
    """预测置信度（0–1 之间）。"""
    ball_id: int
    """被分配到的粒球编号。"""
    distance: float
    """样本到该球心的角度距离（弧度）。"""
    radius: float
    """该球的半径。"""
    inside_radius: bool
    """样本是否在该球半径范围内。"""
    candidate_count: int
    """包含该样本的粒球总数。"""


class GranularBallClassifier:
    """粒球分类器。

    从预计算的 centers.npy、radii.npy、soft_labels.npy 加载分类模型，
    提供单样本 / 批量预测接口。

    Parameters
    ----------
    model_dir : str or Path
        包含 centers.npy, radii.npy, soft_labels.npy 的目录路径。
    """

    def __init__(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)
        self._model_dir = model_dir
        self._centers: np.ndarray = np.load(model_dir / "centers.npy")
        self._radii: np.ndarray = np.load(model_dir / "radii.npy")
        self._soft_labels: np.ndarray = np.load(model_dir / "soft_labels.npy")

        # 从软标签推导硬标签
        self._hard_labels: np.ndarray = self._soft_labels.argmax(axis=1).astype(np.int64)
        self._num_classes: int = self._soft_labels.shape[1]
        self._num_balls: int = len(self._centers)
        self._feature_dim: int = self._centers.shape[1]

        # 尝试加载 metrics 作为元信息
        metrics_path = model_dir / "metrics.json"
        self.meta: dict = {}
        if metrics_path.exists():
            self.meta = json.loads(metrics_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def num_balls(self) -> int:
        """粒球总数。"""
        return self._num_balls

    @property
    def num_classes(self) -> int:
        """分类类别数。"""
        return self._num_classes

    @property
    def feature_dim(self) -> int:
        """输入特征维度。"""
        return self._feature_dim

    @property
    def centers(self) -> np.ndarray:
        """粒球中心矩阵 (num_balls, feature_dim)。"""
        return self._centers

    @property
    def radii(self) -> np.ndarray:
        """粒球半径向量 (num_balls,)。"""
        return self._radii

    @property
    def hard_labels(self) -> np.ndarray:
        """每个粒球的硬标签 (num_balls,)。"""
        return self._hard_labels

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def predict(self, vector: np.ndarray) -> int:
        """对单个特征向量进行分类。

        Parameters
        ----------
        vector : np.ndarray
            形状为 (feature_dim,) 的 1-D 特征向量。

        Returns
        -------
        int
            预测的类别标签。

        Raises
        ------
        ValueError
            输入维度与模型不匹配时抛出。
        """
        self._validate_input(vector, expect_1d=True)
        return int(self._predict_core(vector[None, :])[0])

    def predict_batch(self, matrix: np.ndarray) -> np.ndarray:
        """对一批特征向量进行分类。

        Parameters
        ----------
        matrix : np.ndarray
            形状为 (n_samples, feature_dim) 的 2-D 特征矩阵。

        Returns
        -------
        np.ndarray
            形状为 (n_samples,) 的预测标签，dtype=int64。
        """
        self._validate_input(matrix, expect_1d=False)
        return self._predict_core(matrix)

    def predict_proba(self, vector: np.ndarray) -> np.ndarray:
        """返回单个样本的软标签概率分布。

        找到样本所属粒球，返回该球的软标签作为概率估计。

        Parameters
        ----------
        vector : np.ndarray
            形状为 (feature_dim,) 的 1-D 特征向量。

        Returns
        -------
        np.ndarray
            形状为 (num_classes,) 的概率向量。
        """
        self._validate_input(vector, expect_1d=True)
        _, ball_id = self._find_best_ball(vector[None, :])
        return self._soft_labels[ball_id[0]].copy()

    def predict_detail(self, vector: np.ndarray) -> PredictionDetail:
        """对单个样本进行预测，并返回详细的诊断信息。

        Parameters
        ----------
        vector : np.ndarray
            形状为 (feature_dim,) 的 1-D 特征向量。

        Returns
        -------
        PredictionDetail
            包含标签、置信度、球编号、距离等详细信息的结构体。
        """
        self._validate_input(vector, expect_1d=True)

        vec = vector.astype(np.float32, copy=False)[None, :]
        similarities = np.clip(vec @ self._centers.T, -1.0, 1.0)
        distances: np.ndarray = np.arccos(similarities)  # (1, num_balls)

        in_radius = distances <= self._radii[None, :]  # (1, num_balls)
        candidate_count = int(in_radius.sum())

        best_ball: int
        if candidate_count > 0:
            masked = np.where(in_radius, distances, np.inf)
            best_ball = int(masked.argmin())
        else:
            best_ball = int(distances.argmin())

        best_distance = float(distances[0, best_ball])
        best_radius = float(self._radii[best_ball])
        label = int(self._hard_labels[best_ball])

        # 置信度：基于距离与半径的比值
        if best_radius > 0:
            confidence = float(np.clip(1.0 - best_distance / best_radius, 0.0, 1.0))
        else:
            confidence = 1.0 if best_distance < 1e-6 else 0.0

        return PredictionDetail(
            label=label,
            confidence=confidence,
            ball_id=best_ball,
            distance=best_distance,
            radius=best_radius,
            inside_radius=candidate_count > 0,
            candidate_count=candidate_count,
        )

    def summary(self) -> str:
        """返回分类器的文本摘要。"""
        lines = [
            "=" * 52,
            "  粒球分类器 (Granular Ball Classifier)",
            "=" * 52,
            f"  粒球数量     : {self._num_balls}",
            f"  类别数       : {self._num_classes}",
            f"  特征维度     : {self._feature_dim}",
            f"  半径均值     : {float(self._radii.mean()):.6f} rad",
            f"  半径标准差   : {float(self._radii.std()):.6f} rad",
            f"  最小半径     : {float(self._radii.min()):.6f} rad",
            f"  最大半径     : {float(self._radii.max()):.6f} rad",
        ]
        if self.meta:
            lines.append("-" * 52)
            lines.append("  训练元信息 (metrics.json):")
            for k, v in self.meta.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.6f}")
                else:
                    lines.append(f"    {k}: {v}")
        lines.append("=" * 52)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _validate_input(self, arr: np.ndarray, *, expect_1d: bool) -> None:
        """校验输入维度。"""
        if expect_1d:
            if arr.ndim != 1:
                raise ValueError(
                    f"predict() 期望 1-D 向量，得到 {arr.ndim}-D，"
                    f"形状 {arr.shape}。批量预测请使用 predict_batch()。"
                )
            if arr.shape[0] != self._feature_dim:
                raise ValueError(
                    f"特征维度不匹配：模型期望 {self._feature_dim}，"
                    f"输入为 {arr.shape[0]}。"
                )
        else:
            if arr.ndim != 2:
                raise ValueError(
                    f"predict_batch() 期望 2-D 矩阵，得到 {arr.ndim}-D，"
                    f"形状 {arr.shape}。单样本预测请使用 predict()。"
                )
            if arr.shape[1] != self._feature_dim:
                raise ValueError(
                    f"特征维度不匹配：模型期望 {self._feature_dim}，"
                    f"输入为 {arr.shape[1]}。"
                )

    def _predict_core(self, matrix: np.ndarray) -> np.ndarray:
        """核心预测逻辑（角度距离 → 粒球分配 → 硬标签）。"""
        matrix = matrix.astype(np.float32, copy=False)

        similarities = np.clip(matrix @ self._centers.T, -1.0, 1.0)
        distances: np.ndarray = np.arccos(similarities)  # (n, num_balls)

        in_radius = distances <= self._radii[None, :]
        candidate_counts = in_radius.sum(axis=1)

        nearest_any = distances.argmin(axis=1)
        masked = np.where(in_radius, distances, np.inf)
        nearest_candidate = masked.argmin(axis=1)

        no_candidate = candidate_counts == 0
        nearest_candidate[no_candidate] = nearest_any[no_candidate]

        return self._hard_labels[nearest_candidate]

    def _find_best_ball(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (predicted_labels, assigned_ball_ids)。"""
        similarities = np.clip(matrix @ self._centers.T, -1.0, 1.0)
        distances = np.arccos(similarities)
        in_radius = distances <= self._radii[None, :]
        nearest_any = distances.argmin(axis=1)
        masked = np.where(in_radius, distances, np.inf)
        nearest_candidate = masked.argmin(axis=1)
        no_candidate = in_radius.sum(axis=1) == 0
        nearest_candidate[no_candidate] = nearest_any[no_candidate]
        return self._hard_labels[nearest_candidate], nearest_candidate


# ======================================================================
# 便捷函数
# ======================================================================

def load_classifier(model_dir: Optional[Union[str, Path]] = None) -> GranularBallClassifier:
    """快速加载粒球分类器。

    Parameters
    ----------
    model_dir : str or Path, optional
        模型目录。默认为当前文件所在目录。

    Returns
    -------
    GranularBallClassifier
    """
    if model_dir is None:
        model_dir = Path(__file__).resolve().parent
    return GranularBallClassifier(model_dir)


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    clf = load_classifier()
    print(clf.summary())
    print()
    print("用法示例:")
    print("  from granular_ball_classifier import GranularBallClassifier")
    print("  clf = GranularBallClassifier('粒球聚类结果')")
    print("  label = clf.predict(your_feature_vector)")
