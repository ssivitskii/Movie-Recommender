import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from .data_loader import MovieLensDataLoader
from .preprocessing import DataPreprocessor, train_test_split_ratings, create_genre_matrix, filter_cold_start
from .models import BaseRecommender, UserBasedCF, ItemBasedCF, SVDModel, ContentBasedFilter, HybridRecommender
from .metrics import RecommenderEvaluator

class MovieRecommender:
    """Основной класс рекомендательной системы фильмов"""
    
    def __init__(self, dataset: str = 'ml-100k', model: Union[BaseRecommender, str] = 'svd', 
                 model_params: Dict = None, data_dir: str = 'data', models_dir: str = 'models/trained'):
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_loader = MovieLensDataLoader(dataset, data_dir)
        self.ratings, self.movies = None, None
        self.train_data, self.test_data = None, None
        self.preprocessor = DataPreprocessor()
        self.ratings_matrix, self.genre_matrix = None, None
        self.movie_to_idx, self.idx_to_movie, self.title_to_movie_id = None, None, None
        self.model_params = model_params or {}
        self._init_model(model)
        self._fitted = False
    
    def _init_model(self, model: Union[BaseRecommender, str]):
        if isinstance(model, BaseRecommender):
            self.model, self.model_name = model, type(model).__name__
        else:
            self.model_name = model
            if model == 'svd':
                p = {'n_factors': 100, 'n_epochs': 20, 'lr': 0.005, 'reg': 0.02}
                p.update(self.model_params)
                self.model = SVDModel(**p)
            elif model == 'user_cf':
                p = {'k_neighbors': 50}; p.update(self.model_params)
                self.model = UserBasedCF(**p)
            elif model == 'item_cf':
                p = {'k_neighbors': 50}; p.update(self.model_params)
                self.model = ItemBasedCF(**p)
            elif model == 'content':
                p = {'k_similar': 20}; p.update(self.model_params)
                self.model = ContentBasedFilter(**p)
            elif model == 'hybrid':
                self.model = HybridRecommender()
                self.model.add_model('svd', SVDModel(n_factors=100, n_epochs=20), weight=0.5)
                self.model.add_model('item_cf', ItemBasedCF(k_neighbors=50), weight=0.3)
                self.model.add_model('content', ContentBasedFilter(k_similar=20), weight=0.2)
            else:
                raise ValueError(f"Unknown model: {model}")
    
    def load_data(self, min_user_ratings: int = 5, min_item_ratings: int = 5) -> 'MovieRecommender':
        self.ratings, self.movies = self.data_loader.load_data()
        self.ratings = filter_cold_start(self.ratings, min_user_ratings, min_item_ratings)
        self.title_to_movie_id = dict(zip(self.movies['title'], self.movies['movie_id']))
        return self
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> 'MovieRecommender':
        if self.ratings is None: self.load_data()
        self.train_data, self.test_data = train_test_split_ratings(self.ratings, test_size, random_state)
        self.preprocessor.fit_transform(self.train_data)
        self.ratings_matrix = self.preprocessor.to_dense_matrix(self.train_data)
        movies_in_data = self.movies[self.movies['movie_id'].isin(self.ratings['movie_id'].unique())].reset_index(drop=True)
        self.genre_matrix, self.genre_names, self.movie_to_idx = create_genre_matrix(movies_in_data)
        self.idx_to_movie = {idx: movie for movie, idx in self.movie_to_idx.items()}
        return self
    
    def fit(self, verbose: bool = True) -> 'MovieRecommender':
        if self.ratings_matrix is None: self.prepare_data()
        if isinstance(self.model, (ContentBasedFilter, HybridRecommender)):
            genre_aligned = np.zeros((self.ratings_matrix.shape[1], self.genre_matrix.shape[1]))
            for i in range(self.ratings_matrix.shape[1]):
                mid = self.preprocessor.item_decoder.get(i)
                if mid in self.movie_to_idx: genre_aligned[i] = self.genre_matrix[self.movie_to_idx[mid]]
            self.model.fit(self.ratings_matrix, genre_matrix=genre_aligned, verbose=verbose)
        else:
            self.model.fit(self.ratings_matrix, verbose=verbose)
        self._fitted = True
        return self
    
    def recommend_for_user(self, user_id: int, n: int = 10, exclude_known: bool = True) -> pd.DataFrame:
        if not self._fitted: raise ValueError("Model not fitted")
        uidx = self.preprocessor.user_encoder.get(user_id)
        if uidx is None: raise ValueError(f"Unknown user_id: {user_id}")
        recs = self.model.recommend(uidx, n, exclude_known)
        res = []
        for iidx, rating in recs:
            mid = self.preprocessor.item_decoder.get(iidx)
            if mid:
                info = self.movies[self.movies['movie_id'] == mid].iloc[0]
                res.append({'movie_id': mid, 'title': info['title'], 'genres': info['genres'], 'predicted_rating': round(rating, 2)})
        return pd.DataFrame(res)
    
    def find_similar_movies(self, title: str, n: int = 10) -> pd.DataFrame:
        if not self._fitted: raise ValueError("Model not fitted")
        mid = self.title_to_movie_id.get(title)
        if mid is None:
            matches = [t for t in self.title_to_movie_id.keys() if title.lower() in t.lower()]
            if not matches: raise ValueError(f"Movie not found: {title}")
            mid = self.title_to_movie_id[matches[0]]
        iidx = self.preprocessor.item_encoder.get(mid)
        if iidx is None: raise ValueError("Movie not in training data")
        if isinstance(self.model, ItemBasedCF):
            similar = self.model.find_similar_items(iidx, n)
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(self.ratings_matrix.T)[iidx]
            sims[iidx] = -1
            top = np.argsort(sims)[-n:][::-1]
            similar = [(i, sims[i]) for i in top]
        res = []
        for idx, s in similar:
            mid = self.preprocessor.item_decoder.get(idx)
            if mid:
                info = self.movies[self.movies['movie_id'] == mid]
                if not info.empty:
                    info = info.iloc[0]
                    res.append({'movie_id': mid, 'title': info['title'], 'genres': info['genres'], 'similarity': round(s, 4)})
        return pd.DataFrame(res)

    def evaluate(self, threshold: float = 4.0) -> Dict[str, float]:
        if not self._fitted: raise ValueError("Model not fitted")
        if getattr(self, 'test_data', None) is None or getattr(self, 'ratings_matrix', None) is None:
            raise ValueError("This older model file has no evaluation state; retrain and save it again")
        processed = self.preprocessor.transform(self.test_data)
        tuples = [(int(r['user_idx']), int(r['item_idx']), float(r['rating'])) for _, r in processed.iterrows()]
        evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
        res = evaluator.full_evaluation(self.model, tuples, n_items=self.ratings_matrix.shape[1], threshold=threshold)
        evaluator.print_results(res, f"{self.model_name} Evaluation")
        return res

    def save(self, filepath: str = None):
        if not self._fitted:
            raise ValueError("Model not fitted")
        if filepath is None: filepath = self.models_dir / f"{self.model_name}_{self.dataset}.pkl"
        # Persist evaluation/history state as well as inference state. Only load
        # pickle files that you created or otherwise trust.
        state_names = ('ratings', 'train_data', 'test_data', 'ratings_matrix', 'genre_matrix',
                       'genre_names', 'movie_to_idx', 'idx_to_movie', 'model_params',
                       'data_dir', 'models_dir')
        state = {name: getattr(self, name, None) for name in state_names}
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'preprocessor': self.preprocessor, 'movies': self.movies, 
                         'title_to_movie_id': self.title_to_movie_id, 'model_name': self.model_name,
                         'dataset': self.dataset, 'format_version': 2, 'state': state}, f)

    @classmethod
    def load(cls, filepath: str) -> 'MovieRecommender':
        with open(filepath, 'rb') as f: d = pickle.load(f)
        recommender = cls.__new__(cls)
        recommender.model, recommender.preprocessor, recommender.movies = d['model'], d['preprocessor'], d['movies']
        recommender.title_to_movie_id, recommender.model_name, recommender.dataset = d['title_to_movie_id'], d['model_name'], d['dataset']
        recommender.__dict__.update(d.get('state', {}))
        recommender.models_dir = Path(getattr(recommender, 'models_dir', None) or 'models/trained')
        recommender.data_dir = Path(getattr(recommender, 'data_dir', None) or 'data')
        recommender.data_loader = MovieLensDataLoader(recommender.dataset, str(recommender.data_dir))
        recommender._fitted = True
        return recommender

    def load_model(self, filepath: str) -> 'MovieRecommender':
        """Compatibility API for the prediction CLI; trust the pickle source."""
        self.__dict__.update(type(self).load(filepath).__dict__)
        return self

    def get_user_history(self, user_id: int) -> pd.DataFrame:
        ratings = getattr(self, 'ratings', None)
        if ratings is None:
            ratings = getattr(self, 'train_data', None)
        if ratings is None:
            raise ValueError("This model file does not contain user history")
        history = ratings[ratings.user_id == user_id].merge(self.movies, on='movie_id', how='left')
        order = 'timestamp' if 'timestamp' in history.columns else 'rating'
        return history.sort_values(order, ascending=False).reset_index(drop=True)
