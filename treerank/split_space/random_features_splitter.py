"""
    Random features splitter.
"""
import numpy as np

from .space_splitter import SpaceSplitter

class RandomFeaturesSplitter(SpaceSplitter):
    """Takes a splitter and turns it into a splitter with random features."""

    def __init__(self, splitter_init_not_fit, n_output_feats):
        self._splitter = splitter_init_not_fit
        super(RandomFeaturesSplitter, self).__init__(self._splitter.fact_neg,
                                                     self._splitter.fact_pos)
        self._n_output_feats = n_output_feats
        self._feats = None

    def fit(self, X, y):
        """Fits the space splitter."""
        self._feats = np.random.randint(0, X.shape[1], self._n_output_feats)
        self._splitter.fit(X[:, self._feats], y)
        # We copy the values of the parameters, since they are public and used.
        self.n_pos = self._splitter.n_pos
        self.n_neg = self._splitter.n_neg
        self.best_value = self._splitter.best_value

    def check(self, X):
        """Gives a boolean that checks which elements of X are in the set."""
        return self._splitter.check(X[:, self._feats])

    def plot(self, region, *argv, **kwargs):
        """Plotting is relevant in two-dimensional spaces only."""
        raise Exception("Attempt to plot a random splitter."
                        + "Plotting is relevant in two-dimensional spaces.")

    def cut_region(self, region):
        """Return the regions in and out when the splits cuts the region."""
        raise Exception("Attempt to cut a region with a random splitter."
                        + "Plotting is relevant in two-dimensional spaces.")
