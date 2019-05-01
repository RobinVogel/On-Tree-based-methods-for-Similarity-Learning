"""
    Symmetric random features splitter.
"""
import numpy as np

from .random_features_splitter import RandomFeaturesSplitter

class RandomSymFeaturesSplitter(RandomFeaturesSplitter):
    """Splitter with random features for symmetric splits."""
    def __init__(self, *args, **kwargs):
        self.__dim = None
        super(RandomSymFeaturesSplitter, self).__init__(*args, **kwargs)

    def fit(self, X, y):
        """Fits the space splitter."""
        assert len(X.shape) > 1 and X.shape[1] % 2 == 0
        self.__dim = X.shape[1]
        rand_feats = np.random.randint(0, self.__dim//2, self._n_output_feats)
        self._feats = np.hstack([rand_feats, rand_feats + self.__dim//2])
        self._splitter.fit(X[:, self._feats], y)
        # We copy the values of the parameters, since they are public and used.
        self.n_pos = self._splitter.n_pos
        self.n_neg = self._splitter.n_neg
        self.best_value = self._splitter.best_value
