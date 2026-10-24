"""Full-catalog recommendation with a strict global time split.

Independent of the legacy UI workflow, whose historical random split is retained.
No test interactions, user histories or item popularity enter model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .data_loader import MovieLensDataLoader


def chronological_split(ratings, train_fraction=0.70, validation_fraction=0.15):
    """Keep equal timestamps together, with strict train < validation < test."""
    if not 0 < train_fraction < train_fraction + validation_fraction < 1:
        raise ValueError("Fractions must define three nonempty time intervals")
    stamps = np.sort(ratings.timestamp.unique())
    if len(stamps) < 3:
        raise ValueError("Need at least three distinct timestamps")
    first = int(len(stamps) * train_fraction)
    second = int(len(stamps) * (train_fraction + validation_fraction))
    if not 0 < first < second < len(stamps):
        raise ValueError("Fractions leave an empty interval")
    a, b = int(stamps[first]), int(stamps[second])
    train = ratings[ratings.timestamp < a].copy()
    validation = ratings[(ratings.timestamp >= a) & (ratings.timestamp < b)].copy()
    test = ratings[ratings.timestamp >= b].copy()
    return train, validation, test, {"validation_start": a, "test_start": b}


class SparseRanker:
    """Train-only popularity, sparse item cosine and low-rank SVD baselines."""

    def __init__(self, method, threshold=4.0, neighbors=50, factors=32, seed=42):
        self.method, self.threshold = method, threshold
        self.neighbors, self.factors, self.seed = neighbors, factors, seed

    def fit(self, train):
        self.items = np.sort(train.movie_id.unique())
        self.users = np.sort(train.user_id.unique())
        self.item_index = {value: i for i, value in enumerate(self.items)}
        self.user_index = {value: i for i, value in enumerate(self.users)}
        positive = train[train.rating >= self.threshold]
        rows = positive.user_id.map(self.user_index).to_numpy()
        cols = positive.movie_id.map(self.item_index).to_numpy()
        self.history = sparse.csr_matrix(
            (np.ones(len(positive)), (rows, cols)), shape=(len(self.users), len(self.items))
        )
        self.history.data[:] = 1.0
        counts = np.asarray(self.history.sum(axis=0)).ravel()
        self.popularity = counts / max(1.0, counts.max())
        self.seen = train.groupby("user_id").movie_id.agg(set).to_dict()
        if self.method == "item_cosine":
            item_vectors = normalize(self.history.T, axis=1)
            similarities = (item_vectors @ item_vectors.T).tocsr()
            similarities.setdiag(0)
            similarities.eliminate_zeros()
            # Bounded neighbors; stable tie breaking by item index.
            for row in range(similarities.shape[0]):
                start, stop = similarities.indptr[row:row + 2]
                data = similarities.data[start:stop]
                indexes = similarities.indices[start:stop]
                keep = np.lexsort((indexes, -data))[:self.neighbors]
                remove = np.ones(len(data), dtype=bool)
                remove[keep] = False
                data[remove] = 0
            similarities.eliminate_zeros()
            self.similarity = similarities
        elif self.method == "svd":
            factors = min(self.factors, min(self.history.shape) - 1)
            if factors < 1:
                raise ValueError("SVD requires at least two users and items")
            self.factorizer = TruncatedSVD(n_components=factors, random_state=self.seed)
            self.user_vectors = self.factorizer.fit_transform(self.history)
        elif self.method != "popularity":
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def recommend(self, user_id, k=20):
        index = self.user_index.get(user_id)
        scores = self.popularity.copy()
        if index is not None:
            if self.method == "item_cosine":
                scores = (self.history[index] @ self.similarity).toarray().ravel()
            elif self.method == "svd":
                scores = self.user_vectors[index] @ self.factorizer.components_
            if not np.any(scores > 0):
                scores = self.popularity.copy()
        for item in self.seen.get(user_id, set()):
            scores[self.item_index[item]] = -np.inf
        order = np.lexsort((self.items, -scores))
        return self.items[order[np.isfinite(scores[order])][:k]].tolist()


def ranking_values(predicted, relevant):
    truth = set(relevant)
    hits = np.asarray([item in truth for item in predicted], dtype=float)
    denominator = np.log2(np.arange(2, 12))
    dcg = float((hits[:10] / denominator[:len(hits[:10])]).sum())
    ideal = float((1 / denominator[:min(10, len(truth))]).sum())
    return {"ndcg_at_10": dcg / ideal if ideal else 0.0,
            "recall_at_20": float(hits[:20].sum() / len(truth)) if truth else 0.0}


def evaluate(ranker, heldout, bootstrap_samples=500, seed=42):
    positives = heldout[heldout.rating >= ranker.threshold]
    truth = positives.groupby("user_id").movie_id.agg(set).to_dict()
    rows, recommended = [], set()
    durations = []
    for user, items in sorted(truth.items()):
        relevant = items - ranker.seen.get(user, set())
        if not relevant:
            continue
        started = time.perf_counter()
        predicted = ranker.recommend(user)
        durations.append((time.perf_counter() - started) * 1000)
        recommended.update(predicted[:10])
        n_history = len(ranker.seen.get(user, set()))
        rows.append({"user_id": int(user), **ranking_values(predicted, relevant),
                     "history": n_history,
                     "cold_item_relevant": bool(set(relevant) - set(ranker.item_index)),
                     "relevant": len(relevant),
                     "unavailable_relevant": len(set(relevant) - set(ranker.item_index))})
    if not rows:
        raise ValueError("No held-out positive users to evaluate")
    frame = pd.DataFrame(rows)
    metric_names = ["ndcg_at_10", "recall_at_20"]
    rng = np.random.default_rng(seed)
    means = frame[metric_names].to_numpy()
    estimates = np.asarray([means[rng.integers(0, len(means), len(means))].mean(axis=0)
                            for _ in range(bootstrap_samples)])
    result = {name: float(frame[name].mean()) for name in metric_names}
    result.update({"users": len(frame), "coverage_at_10": len(recommended) / len(ranker.items),
                   "train_catalog_items": len(ranker.items),
                   "unavailable_relevant_fraction": float(frame.unavailable_relevant.sum() / frame.relevant.sum()),
                   "latency_ms_p50": float(np.percentile(durations, 50)),
                   "latency_ms_p95": float(np.percentile(durations, 95)),
                   "bootstrap_95_ci": {name: np.quantile(estimates[:, i], [.025, .975]).tolist()
                                       for i, name in enumerate(metric_names)}})
    masks = {"cold_user": frame.history.eq(0),
             "short_history_1_to_9": frame.history.between(1, 9),
             "history_10_plus": frame.history.ge(10),
             "has_unavailable_relevant_item": frame.cold_item_relevant}
    result["slices"] = {name: {"users": int(mask.sum()),
                               **{metric: float(frame.loc[mask, metric].mean()) if mask.any() else None
                                  for metric in metric_names}}
                        for name, mask in masks.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="reports/temporal_ml100k.json")
    args = parser.parse_args()
    loader = MovieLensDataLoader("ml-100k", args.data_dir)
    ratings = loader.load_ratings()
    train, validation, test, cutoffs = chronological_split(ratings)
    # Parameters are fixed in source before evaluation. Validation is reported,
    # then every baseline refits on train+validation for one future test period.
    report = {"dataset": "MovieLens 100K", "source": loader.config["url"],
              "input_sha256": hashlib.sha256((loader.raw_dir / "ml-100k/u.data").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol": {"split": "global chronological 70/15/15 of unique timestamps",
                           "candidate_catalog": "items observed before each evaluation cutoff",
                           "relevance": "rating >= 4; already rated items excluded",
                           "aggregation": "macro mean over users with held-out positives",
                           "missing_relevant": "unseen catalog targets remain in recall/NDCG denominator",
                           "bootstrap": "500 resamples of users, seed 42; conditional on fixed models",
                           "parameters": {"item_neighbors": 50, "svd_factors": 32, "seed": 42},
                           "selection": "fixed parameters; no test selection", **cutoffs},
              "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "scipy": scipy.__version__,
                          "scikit_learn": sklearn.__version__, "platform": platform.platform()},
              "models": {}}
    for method in ["popularity", "item_cosine", "svd"]:
        ranker = SparseRanker(method).fit(train)
        val_metrics = evaluate(ranker, validation)
        started = time.perf_counter()
        ranker = SparseRanker(method).fit(pd.concat([train, validation]))
        fit_seconds = time.perf_counter() - started
        report["models"][method] = {"validation": val_metrics, "test": evaluate(ranker, test),
                                     "test_fit_seconds": fit_seconds}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: model["test"] for name, model in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
