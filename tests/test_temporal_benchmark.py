import numpy as np
import pandas as pd
import pytest

from src.temporal_benchmark import SparseRanker, chronological_split, evaluate, ranking_values


def fixture():
    return pd.DataFrame({"user_id": [1, 1, 2, 2, 3, 3, 1, 2, 4, 1],
                         "movie_id": [1, 2, 1, 3, 2, 3, 4, 4, 4, 5],
                         "rating": [5] * 10, "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


def test_timestamp_groups_stay_together():
    data = pd.concat([fixture(), fixture()])
    train, validation, test, _ = chronological_split(data)
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert len(train) + len(validation) + len(test) == len(data)


def test_seen_filter_and_unseen_catalog():
    train = fixture().iloc[:6]
    model = SparseRanker("popularity").fit(train)
    assert set(model.recommend(1)).isdisjoint({1, 2})
    assert 4 not in model.items
    assert model.recommend(99) == [1, 2, 3]
    report = evaluate(model, fixture().iloc[6:], bootstrap_samples=5)
    assert report["unavailable_relevant_fraction"] == 1.0
    assert report["recall_at_20"] == 0.0
    assert report["slices"]["cold_user"]["users"] == 1


@pytest.mark.parametrize("method", ["popularity", "item_cosine", "svd"])
def test_train_only_rankers_are_deterministic(method):
    data = fixture().iloc[:6]
    a = SparseRanker(method, factors=2).fit(data)
    b = SparseRanker(method, factors=2).fit(data)
    assert a.recommend(1) == b.recommend(1)
    assert set(a.recommend(1)).isdisjoint(a.seen[1])


def test_metric_denominator_includes_unretrievable_relevance():
    values = ranking_values([10], {10, 999})
    assert values["recall_at_20"] == 0.5
    assert np.isclose(values["ndcg_at_10"], 1 / (1 + 1 / np.log2(3)))
