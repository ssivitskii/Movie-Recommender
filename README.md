# Movie Recommender

**A comparison of collaborative, content-based and hybrid recommendation methods on MovieLens.**

[Source](https://github.com/ssivitskii/Movie-Recommender) · [Issues](https://github.com/ssivitskii/Movie-Recommender/issues) · [Contributing](CONTRIBUTING.md)

## What it does

Generates recommendations for existing users and finds similar movies. The repository exposes a Python interface, command-line scripts and a Streamlit app.

| Model key | Approach |
| --- | --- |
| `svd` | Matrix factorization |
| `user_cf` | User-based collaborative filtering |
| `item_cf` | Item-based collaborative filtering |
| `content` | Content-based similarity |
| `hybrid` | Weighted combination of SVD, item-based and content models |

**Stack:** Python · NumPy · pandas · SciPy · scikit-learn · Streamlit

## Run locally

Use Python 3.11:

```bash
git clone https://github.com/ssivitskii/Movie-Recommender.git
cd Movie-Recommender
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,web]"
```

The data loader downloads MovieLens on first use; network access is required. Supported datasets are `ml-100k`, `ml-1m` and `ml-latest-small`. Start with `ml-100k` because some implementations construct dense matrices.

```bash
# Train and evaluate
python -m src.train --model svd --dataset ml-100k --epochs 20 --evaluate

# Compare implementations; this trains models again
python -m src.evaluate --model all --dataset ml-100k

# Open the web interface
python -m streamlit run streamlit_app.py
```

### Python interface

```python
from src.recommender import MovieRecommender

recommender = MovieRecommender(dataset="ml-100k", model="svd")
recommender.fit()
recommendations = recommender.recommend_for_user(user_id=1, n=10)
print(recommendations)

# Select model="hybrid" to train the weighted hybrid implementation.
```

## Evaluation and limitations

```bash
python -m pytest
```

The evaluation code computes rating errors (RMSE and MAE) and ranking metrics including Precision@K, Recall@K and NDCG@K. Record the dataset, filtering, split, model parameters and relevance threshold with each experiment. No versioned benchmark report is included, so this README does not advertise fixed quality or latency numbers.

The default data split is random, not chronological. The hybrid uses fixed weights (SVD 0.5, item-based 0.3, content-based 0.2). Dense matrices limit scalability, and cold-start handling is not a production solution.

The saved-model CLI paths need further work: `src.predict --model-path` refers to a missing `load_model` method, and loading does not restore all state used by evaluation. Use the in-memory training workflow above. `--show-history` and `--similar-to` in the prediction script are also incomplete.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/recommender.py` | Python orchestration interface |
| `src/models.py` | Recommendation algorithms |
| `src/data_loader.py` | MovieLens loading and preparation |
| `src/metrics.py` | Evaluation metrics |
| `src/train.py`, `src/evaluate.py` | Training and comparison CLI |
| `streamlit_app.py` | Web interface |
| `notebooks/01_eda.py` | Exploratory analysis script |
| `tests/` | Automated tests |

## License and data

Code: [MIT](LICENSE). MovieLens datasets are provided by [GroupLens](https://grouplens.org/datasets/movielens/) under their own usage terms.
