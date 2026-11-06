from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Абсолютный импорт работает, даже если запускать файл из другого каталога
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from partition_clustering import add_noise, assignment_frame, cluster, load_and_prepare


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("kmeans", "kmedoids")
K_VALUES = tuple(range(3, 10))
NOISE_FRACTIONS = (0.00, 0.01, 0.03, 0.05, 0.10)
COLORS = plt.get_cmap("tab10")

# Стабильные подписи с кириллицей на Windows
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def noise_label(value: float) -> str:
    return f"{int(round(value * 100))}%"


def plot_error_curves(metrics: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharey=False)
    for axis, algorithm in zip(axes, ALGORITHMS):
        subset = metrics[metrics.algorithm == algorithm]
        for index, noise in enumerate(NOISE_FRACTIONS):
            rows = subset[np.isclose(subset.noise_fraction, noise)].sort_values("k")
            axis.plot(rows.k, rows.error_sse, marker="o", linewidth=2.1,
                      color=COLORS(index), label=f"шум {noise_label(noise)}")
        axis.set_title("k-Means" if algorithm == "kmeans" else "k-Medoids", fontweight="bold")
        axis.set_xlabel("Число кластеров k")
        axis.set_ylabel("Ошибка кластеризации (SSE)")
        axis.set_xticks(K_VALUES)
        axis.legend(fontsize=8)
    fig.suptitle("Зависимость ошибки кластеризации от k и доли зашумленных объектов",
                 fontsize=15, fontweight="bold", color="#17324d")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_clean_all_k(projected: np.ndarray, runs: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, len(K_VALUES), figsize=(18, 6.2), sharex=True, sharey=True)
    for row, algorithm in enumerate(ALGORITHMS):
        for column, k in enumerate(K_VALUES):
            axis = axes[row, column]
            labels = runs[(algorithm, 0.0, k)][0]
            axis.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="tab10", s=8, alpha=0.72,
                         linewidths=0)
            axis.set_title(f"k={k}", fontsize=10)
            if column == 0:
                axis.set_ylabel(("k-Means" if row == 0 else "k-Medoids") + "\nPC2")
            if row == 1:
                axis.set_xlabel("PC1")
    fig.suptitle("Кластеры без шума: проекция стандартизованных данных на первые две компоненты PCA",
                 fontsize=14, fontweight="bold", color="#17324d")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_noise_scatter(prepared, noisy_sets: dict, runs: dict, k: int, output: Path) -> None:
    pca = PCA(n_components=2, random_state=42).fit(prepared.coordinates)
    fig, axes = plt.subplots(len(NOISE_FRACTIONS), 2, figsize=(11, 17), sharex=True, sharey=True)
    for row, noise in enumerate(NOISE_FRACTIONS):
        coordinates, noisy_mask = noisy_sets[noise]
        projection = pca.transform(coordinates)
        for column, algorithm in enumerate(ALGORITHMS):
            axis = axes[row, column]
            labels = runs[(algorithm, noise, k)][0]
            axis.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", s=11,
                         alpha=0.72, linewidths=0)
            if noisy_mask.any():
                axis.scatter(projection[noisy_mask, 0], projection[noisy_mask, 1], facecolors="none",
                             edgecolors="#111827", s=35, linewidths=0.7, label="изменённые объекты")
            axis.set_title(f"{'k-Means' if algorithm == 'kmeans' else 'k-Medoids'}, шум {noise_label(noise)}")
            axis.set_ylabel("PC2")
            if row == len(NOISE_FRACTIONS) - 1:
                axis.set_xlabel("PC1")
            if noisy_mask.any() and column == 1:
                axis.legend(fontsize=7, loc="best")
    fig.suptitle(f"Принадлежность к кластерам при k={k}; чёрным контуром отмечены зашумленные объекты",
                 fontsize=14, fontweight="bold", color="#17324d")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_relative_growth(metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    width = 0.075
    x = np.arange(len(K_VALUES), dtype=float)
    bar_index = 0
    palette = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    for algorithm, hatch in zip(ALGORITHMS, ("", "//")):
        base = metrics[(metrics.algorithm == algorithm) & np.isclose(metrics.noise_fraction, 0)].set_index("k").error_sse
        for color, noise in zip(palette, NOISE_FRACTIONS[1:]):
            current = metrics[(metrics.algorithm == algorithm) & np.isclose(metrics.noise_fraction, noise)].set_index("k").error_sse
            growth = ((current / base) - 1) * 100
            offset = (bar_index - 3.5) * width
            ax.bar(x + offset, growth.reindex(K_VALUES), width, color=color, hatch=hatch,
                   edgecolor="#334155", linewidth=0.35,
                   label=f"{'k-Means' if algorithm == 'kmeans' else 'k-Medoids'}, {noise_label(noise)}")
            bar_index += 1
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(x, K_VALUES)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Рост SSE относительно данных без шума, %")
    ax.set_title("Чувствительность алгоритмов к шуму", fontweight="bold", color="#17324d")
    ax.legend(ncol=2, fontsize=7.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "customers.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--noise-scale", type=float, default=3.0)
    parser.add_argument("--scatter-k", type=int, default=5)
    args = parser.parse_args()
    if args.scatter_k not in K_VALUES:
        raise ValueError(f"scatter-k must be one of {K_VALUES}")

    prepared = load_and_prepare(args.data)
    output = args.output_dir
    assignments = output / "assignments"
    output.mkdir(parents=True, exist_ok=True)
    assignments.mkdir(parents=True, exist_ok=True)
    noisy_sets = {
        noise: add_noise(prepared.coordinates, noise, random_state=args.random_state, scale=args.noise_scale)
        for noise in NOISE_FRACTIONS
    }
    rows: list[dict] = []
    runs: dict[tuple[str, float, int], tuple[np.ndarray, float]] = {}
    for noise in NOISE_FRACTIONS:
        coordinates, noisy_mask = noisy_sets[noise]
        for algorithm in ALGORITHMS:
            for k in K_VALUES:
                print(f"{algorithm}: noise={noise:.2f}, k={k}", file=sys.stderr, flush=True)
                labels, centers, error, iterations = cluster(
                    coordinates, algorithm, k, random_state=args.random_state
                )
                silhouette = float(silhouette_score(coordinates, labels))
                rows.append({
                    "algorithm": algorithm, "noise_fraction": noise,
                    "noisy_rows": int(noisy_mask.sum()), "k": k,
                    "error_sse": error, "silhouette": silhouette,
                    "iterations": iterations,
                    "min_cluster_size": int(np.bincount(labels, minlength=k).min()),
                    "max_cluster_size": int(np.bincount(labels, minlength=k).max()),
                })
                runs[(algorithm, noise, k)] = (labels, error)
                filename = f"{algorithm}_noise_{int(noise * 100):02d}_k_{k}.csv"
                assignment_frame(prepared, coordinates, labels, noisy_mask).to_csv(assignments / filename, index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "experiment_metrics.csv", index=False)
    pca = PCA(n_components=2, random_state=args.random_state).fit(prepared.coordinates)
    plot_error_curves(metrics, output / "error_by_k.png")
    plot_clean_all_k(pca.transform(prepared.coordinates), runs, output / "clusters_clean_all_k.png")
    plot_noise_scatter(prepared, noisy_sets, runs, args.scatter_k, output / "clusters_by_noise_k5.png")
    plot_relative_growth(metrics, output / "noise_sensitivity.png")

    best = []
    for algorithm in ALGORITHMS:
        for noise in NOISE_FRACTIONS:
            subset = metrics[(metrics.algorithm == algorithm) & np.isclose(metrics.noise_fraction, noise)]
            row = subset.loc[subset.silhouette.idxmax()].to_dict()
            best.append(row)
    summary = {
        "dataset": "customers",
        "rows": len(prepared.source),
        "features": prepared.feature_names,
        "preprocessing": "median imputation and z-score standardization",
        "algorithms": list(ALGORITHMS),
        "k_values": list(K_VALUES),
        "noise_fractions": list(NOISE_FRACTIONS),
        "noise_scale": args.noise_scale,
        "random_state": args.random_state,
        "error": "sum of squared Euclidean distances to centroid or medoid (SSE)",
        "best_by_silhouette": best,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()