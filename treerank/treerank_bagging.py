"""
    Implementation of bagged models
"""
import multiprocessing
import numpy as np
from joblib import Parallel, delayed

class BaggedModel:
    """
        Implements a bagged model.

        Used methods:
        * fit, get_rank_cell
    """

    def __init__(self, n_sample, card_sample, model_instantiator):
        self.n_sample = n_sample
        self.card_sample = card_sample
        self.parallel = True
        # private
        self._model_instantiator = model_instantiator
        self._fitted_models = list()

    def fit(self, X, y):
        """Fits several trees."""
        samples = [self._get_random_sample(X, y)
                   for _ in range(self.n_sample)]

        if self.parallel:
            pool = Parallel(n_jobs=multiprocessing.cpu_count()-10)
            self._fitted_models = pool(
                delayed(self._get_fitted_model)(X_samp, y_samp)
                for X_samp, y_samp in samples
            )
        else:
            self._fitted_models = [
                self._get_fitted_model(X_samp, y_samp)
                for X_samp, y_samp in samples
            ]

    def median_rank(self, X):
        """
        Returns the median of the rank of the cells of the X's.

        Which with that notation just means that lots of cells
        will have the same rank.
        """
        rank_values = list()

        if self.parallel:
            pool = Parallel(n_jobs=multiprocessing.cpu_count()//2)
            rank_values = pool(delayed(lambda x: x.get_rank_cell(X))(model)
                               for model in self._fitted_models)
        else:
            for model in self._fitted_models:
                rank_values.append(model.get_rank_cell(X))
        rank_array = np.array(rank_values).transpose()
        return np.mean(rank_array, axis=1)
        # return np.median(rank_array, axis=1)

    def score(self, X):
        """A score, derived from the median rank."""
        return - self.median_rank(X)

    def _get_random_sample(self, X, y):
        ind_sample = np.random.randint(0, X.shape[0], self.card_sample)
        return X[ind_sample], y[ind_sample]

    def _get_fitted_model(self, X_resampled, y_resampled):
        cur_model = self._model_instantiator()
        cur_model.fit(X_resampled, y_resampled)
        return cur_model
