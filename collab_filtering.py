from typing import Tuple
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

ITEM_ITEM = 0
USER_USER = 1


class CollaborativeFiltering:
    def __init__(self):
        self._ratings = None
        self._normalized_ratings = None
        self._norms = None
        self.fitted = False

    def fit(self, ratings: np.ndarray) -> None:
        item_item, user_user = ratings, ratings.T

        item_item_normalized = item_item - np.nanmean(item_item, axis=1, keepdims=True)
        item_item_normalized[np.isnan(item_item_normalized)] = 0.

        user_user_normalized = user_user - np.nanmean(user_user, axis=1, keepdims=True)
        user_user_normalized[np.isnan(user_user_normalized)] = 0.

        item_item_norms = np.linalg.norm(item_item_normalized, axis=1)
        user_user_norms = np.linalg.norm(user_user_normalized, axis=1)

        self._ratings = (item_item, user_user)
        self._normalized_ratings = (item_item_normalized, user_user_normalized)
        self._norms = (item_item_norms, user_user_norms)

        self.fitted = True

    def predict(self, target: Tuple[int, int], query_type: int, k: int) -> Decimal:
        if self.fitted is False:
            raise ValueError(f"This instance of {self.__class__} has not been fitted yet")

        row_idx, column_idx = target
        if query_type == USER_USER:
            column_idx, row_idx = row_idx, column_idx

        ratings = self._ratings[query_type]
        if not np.isnan(ratings[row_idx, column_idx]):
            raise ValueError(f"Rating for item ({row_idx}, {column_idx}) already exists")

        normalized_ratings = self._normalized_ratings[query_type]
        norms = self._norms[query_type]

        similar = np.dot(normalized_ratings[row_idx], normalized_ratings.T) / (norms[row_idx] * norms)

        candidates = np.arange(0, len(similar))
        candidates = candidates[~np.isnan(ratings[:, column_idx])]
        candidates = candidates[similar[candidates] > 0.]

        k_nearest = candidates[np.argsort(-similar[candidates])[:k]]

        similar_k_nearest = similar[k_nearest]
        ratings_k_nearest = ratings[:, column_idx][k_nearest]
        predicted_rating = np.dot(ratings_k_nearest, similar_k_nearest) / np.sum(similar_k_nearest)

        return Decimal(Decimal(predicted_rating).quantize(Decimal('.001'), rounding=ROUND_HALF_UP))
