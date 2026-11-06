from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


IDENTIFIER_COLUMNS = ("Row", "CustomerId")


@dataclass
class PreparedData:
    source: pd.DataFrame
    coordinates: np.ndarray
    feature_names: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler


def load_and_prepare(path: Path) -> PreparedData:
    # utf-8-sig безопасен и для обычного UTF-8, и для файлов с BOM (Excel)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    features = frame.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in frame.columns])
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise ValueError(f"Поддерживаются только числовые признаки. Нечисловые столбцы: {non_numeric}")
    if len(frame) < 2:
        raise ValueError("Набор данных должен содержать не менее двух объектов")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    values = scaler.fit_transform(imputer.fit_transform(features))
    return PreparedData(frame, values, list(features.columns), imputer, scaler)


def add_noise(
    coordinates: np.ndarray,
    fraction: float,
    *,
    random_state: int = 42,
    scale: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= fraction <= 1:
        raise ValueError("Доля зашумленных объектов должна находиться в диапазоне от 0 до 1")
    result = np.array(coordinates, dtype=float, copy=True)
    noisy = np.zeros(len(result), dtype=bool)
    count = int(round(len(result) * fraction))
    if count == 0:
        return result, noisy
    rng = np.random.default_rng(random_state)
    rows = rng.choice(len(result), size=count, replace=False)
    for row in rows:
        coordinate_count = int(rng.integers(1, result.shape[1] + 1))
        columns = rng.choice(result.shape[1], size=coordinate_count, replace=False)
        result[row, columns] += rng.normal(0.0, scale, size=coordinate_count)
    noisy[rows] = True
    return result, noisy


class KMedoids:

    def __init__(self, n_clusters: int, *, max_iter: int = 100, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def _initialize(self, distances: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n = len(distances)
        medoids = [int(rng.integers(n))]
        while len(medoids) < self.n_clusters:
            closest_sq = np.min(distances[:, medoids], axis=1) ** 2
            closest_sq[medoids] = 0
            total = closest_sq.sum()
            if total == 0:
                candidates = np.setdiff1d(np.arange(n), medoids)
                medoids.append(int(rng.choice(candidates)))
            else:
                medoids.append(int(rng.choice(n, p=closest_sq / total)))
        return np.asarray(medoids, dtype=int)

    def fit(self, x: np.ndarray) -> "KMedoids":
        if not 1 <= self.n_clusters <= len(x):
            raise ValueError("Число кластеров должно быть от 1 до количества объектов")
        distances = pairwise_distances(x, metric="euclidean")
        medoids = self._initialize(distances)
        for iteration in range(1, self.max_iter + 1):
            labels = np.argmin(distances[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for cluster in range(self.n_clusters):
                members = np.flatnonzero(labels == cluster)
                if len(members) == 0:
                    nearest = np.min(distances[:, medoids], axis=1)
                    nearest[medoids] = -1
                    new_medoids[cluster] = int(np.argmax(nearest))
                    continue
                within = distances[np.ix_(members, members)]
                new_medoids[cluster] = int(members[np.argmin(within.sum(axis=1))])
            self.n_iter_ = iteration
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids
        labels = np.argmin(distances[:, medoids], axis=1)
        self.medoid_indices_ = medoids
        self.labels_ = labels
        self.inertia_ = float(np.sum(distances[np.arange(len(x)), medoids[labels]] ** 2))
        return self


def cluster(
    coordinates: np.ndarray,
    algorithm: str,
    k: int,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    if not 2 <= k <= len(coordinates):
        raise ValueError("Параметр k должен быть от 2 до количества объектов")
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=random_state)
        labels = model.fit_predict(coordinates)
        return labels, model.cluster_centers_, float(model.inertia_), int(model.n_iter_)
    if algorithm == "kmedoids":
        model = KMedoids(n_clusters=k, random_state=random_state).fit(coordinates)
        assert model.labels_ is not None and model.medoid_indices_ is not None and model.inertia_ is not None
        return model.labels_, coordinates[model.medoid_indices_], model.inertia_, model.n_iter_
    raise ValueError(f"Неизвестный алгоритм: {algorithm}")


def assignment_frame(
    prepared: PreparedData,
    coordinates: np.ndarray,
    labels: np.ndarray,
    noisy: np.ndarray,
) -> pd.DataFrame:
    identity = prepared.source[[c for c in IDENTIFIER_COLUMNS if c in prepared.source]].reset_index(drop=True)
    scaled = pd.DataFrame(coordinates, columns=[f"z_{name}" for name in prepared.feature_names])
    return pd.concat([identity, scaled], axis=1).assign(noisy=noisy, cluster=labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition clustering with k-means or k-medoids")
    parser.add_argument("dataset", type=Path, help="CSV dataset")
    parser.add_argument("--algorithm", choices=("kmeans", "kmedoids"), default="kmeans")
    parser.add_argument("--k", type=int, required=True, help="number of clusters")
    parser.add_argument("--noise", type=float, default=0.0, help="fraction of noisy rows, from 0 to 1")
    parser.add_argument("--noise-scale", type=float, default=3.0, help="noise standard deviation after scaling")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, help="CSV with coordinates and assigned cluster")
    args = parser.parse_args()

    prepared = load_and_prepare(args.dataset)
    coordinates, noisy = add_noise(
        prepared.coordinates, args.noise, random_state=args.random_state, scale=args.noise_scale
    )
    labels, centers, error, iterations = cluster(
        coordinates, args.algorithm, args.k, random_state=args.random_state
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        assignment_frame(prepared, coordinates, labels, noisy).to_csv(args.output, index=False)
    summary = {
        "algorithm": args.algorithm,
        "k": args.k,
        "rows": len(coordinates),
        "features": prepared.feature_names,
        "noise_fraction": args.noise,
        "noisy_rows": int(noisy.sum()),
        "clustering_error_sse": error,
        "iterations": iterations,
        "cluster_sizes": np.bincount(labels, minlength=args.k).tolist(),
        "centers_or_medoids": centers.tolist(),
        "assignment_file": str(args.output) if args.output else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()