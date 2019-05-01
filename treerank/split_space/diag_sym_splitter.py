"""
    Symmetric diagonal splitter.
"""
import numpy as np

from .coordinate_splitter import CoordinateSplitter

class DiagSymSplitter(CoordinateSplitter):
    """
        Coordinate space splitter with symmetry and optional special diag that
        take into account the fact that we want it to keep a diagonal property.
    """

    def __init__(self, fact_neg, fact_pos):
        self.only_upper = None
        super(DiagSymSplitter, self).__init__(fact_neg, fact_pos)
        # Do we impose the diagonal to be in the set ?
        self.__diagonal_in = False

    def check(self, X):
        """Gives a boolean that checks which elements of X are in the set."""
        X_tr = self.__basis_change(X)
        return super(DiagSymSplitter, self).check(X_tr)

    def _space_fit(self, X, y):
        assert len(X.shape) > 1 and X.shape[1] % 2 == 0
        dim = X.shape[1]
        best_split = {"best_value": -float("inf")}
        X_tr = self.__basis_change(X)
        if self.__diagonal_in:
            cand_dims = range(dim//2)
            self.only_upper = True
        else:
            cand_dims = range(dim)
        for d in cand_dims:
            cur_split = self._line_fit(X_tr[:, d], y)
            if cur_split["best_value"] > best_split["best_value"]:
                best_split = cur_split
                best_split["cut_dim"] = d
        return best_split

    # --------------- Private methods ---------------

    @staticmethod
    def __basis_change(X):
        """Change the basis of the data."""
        assert len(X.shape) > 1 and X.shape[1] % 2 == 0
        dim = X.shape[1]
        X_1, X_2 = X[:, :(dim//2)], X[:, (dim//2):]
        return np.hstack([np.abs(X_1 - X_2), X_1 + X_2])/np.sqrt(2)
