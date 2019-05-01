"""
    Implementation of a similarity TreeRank algorithm
"""
from treerank.utils import get_random_pairs
from treerank.treerank import TreeRank
from treerank.treerank_bagging import BaggedModel

class SimTreeRank(TreeRank):
    """Similarity ranking tree. Score function is on the pairs."""
    def __init__(self, splitter_instantiator, depth,
                 n_pairs=None, change_emp_measure=True, alpha_constraint=None):
        # pylint: disable=too-many-arguments
        self.n_pairs = n_pairs
        super(SimTreeRank, self).__init__(
            splitter_instantiator, depth,
            change_emp_measure=change_emp_measure,
            alpha_constraint=alpha_constraint)

    def fit_pairs(self, X_pairs, z, spec_left_fit=False):
        """Fits the pairs provided to the algorithm."""
        self.n_pairs = X_pairs.shape[0]
        super(SimTreeRank, self).fit(X_pairs, z, spec_left_fit=spec_left_fit)

    def fit(self, X, y, spec_left_fit=False):
        assert self.n_pairs is not None
        X_pairs, z = get_random_pairs(X, y, self.n_pairs)
        self.fit_pairs(X_pairs, z, spec_left_fit=spec_left_fit)

class SimBagging(BaggedModel):
    """Bagging for similarity treerank."""
    def _get_random_sample(self, X, y):
        return get_random_pairs(X, y, self.card_sample)

    def _get_fitted_model(self, X_resampled, y_resampled):
        cur_model = self._model_instantiator()
        cur_model.fit_pairs(X_resampled, y_resampled)
        return cur_model
