"""
    Splits the space using a standard CART with our criterion.
"""
import numpy as np
from treerank.treerank import TreeRank
from treerank.split_space.space_splitter import SpaceSplitter
from treerank.split_space.random_features_splitter \
        import RandomFeaturesSplitter
from treerank.split_space.coordinate_splitter import CoordinateSplitter

class CartSplitter(SpaceSplitter):
    r"""
        Implements the LeafRank algorithm of [1].
    """
    def __init__(self, fact_neg, fact_pos, depth, randomize_features=None):
        super(CartSplitter, self).__init__(fact_neg, fact_pos)
        self._l_parts, self._r_parts = [list()]*2

        self._treerank = None
        self._define_treerank(depth, randomize_features)
        self._treerank.set_fixed_factors(fact_neg, fact_pos)

    def fit(self, X, y):
        """
            Fits CartSplitter to the data.

            :param fact_neg:    Weight on negative local density.
            :param fact_pos:    Weight on positive local density.
            :param X:           Position of each point.
            :param y:           Binary class of the point, in {0,1}.

            :return:            None.
        """
        # Run TreeRank to obtain a partition of the space, with a full tree.
        self._treerank.fit(X, y)
        self._find_split_from_fitted_tree()

    def check(self, X):
        """Gives a boolean that checks which elements of X are in the set."""
        part_vals = np.around(
            self._treerank.score(X)*2**self._treerank.depth).astype(int)
        l_parts_nr = [p for p, _ in self._l_parts] # l_parts no ratio
        return np.array([p in l_parts_nr for p in part_vals])

    def plot(self, region, *argv, **kwargs):
        """Plots the region delimited by the splitter."""
        raise Exception("TODO: Not yet implemented.")

    def cut_region(self, region):
        """Return the regions in and out when the splits cuts the region."""
        raise Exception("TODO: Not yet implemented.")

    # --------------- Protected methods ---------------

    def _find_split_from_fitted_tree(self):
        """Finds a split from a fitted tree."""
        # Obtain the list of tuples (parition_id, (alpha, beta)) from tree .
        partitions = self._treerank.get_leaf_partitions()

        # Sort the partitions following the partition-based splitting rule.
        self._maximize_crit_from_parts(partitions)

        for _, (p_n_neg, p_n_pos) in self._l_parts:
            self.n_neg += p_n_neg
            self.n_pos += p_n_pos

    def _define_treerank(self, depth, randomize_features):
        if randomize_features is None:
            split_inst = CoordinateSplitter
        else:
            split_inst = lambda f_n, f_p: RandomFeaturesSplitter(
                CoordinateSplitter(f_n, f_p), randomize_features)
        self._treerank = TreeRank(split_inst, depth, change_emp_measure=False)
