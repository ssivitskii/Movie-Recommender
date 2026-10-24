import numpy as np
import pandas as pd

from src.recommender import MovieRecommender


def test_saved_model_restores_predictions_history_and_evaluation(tmp_path):
    recommender = MovieRecommender(model='svd', model_params={'n_factors': 2, 'n_epochs': 1},
                                   data_dir=str(tmp_path / 'data'), models_dir=str(tmp_path / 'models'))
    recommender.ratings = pd.DataFrame([
        {'user_id': user, 'movie_id': (item + user - 2) % 5 + 1, 'rating': float(2 + (user + item) % 4), 'timestamp': user * 10 + item}
        for user in range(1, 5) for item in range(1, 6)
    ])
    recommender.movies = pd.DataFrame({'movie_id': range(1, 6),
                                      'title': [f'Movie {i}' for i in range(1, 6)],
                                      'genres': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy']})
    recommender.title_to_movie_id = dict(zip(recommender.movies.title, recommender.movies.movie_id))
    recommender.fit(verbose=False)
    path = tmp_path / 'model.pkl'
    recommender.save(path)
    loaded = MovieRecommender.load(path)
    pd.testing.assert_frame_equal(loaded.recommend_for_user(1), recommender.recommend_for_user(1))
    pd.testing.assert_frame_equal(loaded.get_user_history(1), recommender.get_user_history(1))
    pd.testing.assert_frame_equal(loaded.test_data, recommender.test_data)
    np.testing.assert_array_equal(loaded.ratings_matrix, recommender.ratings_matrix)
    assert loaded.evaluate() == recommender.evaluate()
