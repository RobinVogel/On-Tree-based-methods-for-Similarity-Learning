"""
    Symmetric CART for Leafrank.
"""
from treerank.cart_splitter import CartSplitter
from treerank.treerank_similarity import SimTreeRank

from treerank.split_space.random_sym_features_splitter \
        import RandomSymFeaturesSplitter
from treerank.split_space.cross_sym_splitter import CrossSymSplitter
from treerank.split_space.diag_sym_splitter import DiagSymSplitter

class CartSymSplitter(CartSplitter):
    r"""
        Implements the CartSplitter algorithm of [1].

        randomize_features is an integer.
    """
    def __init__(self, fact_neg, fact_pos, depth, randomize_features=None):
        self._define_treerank(depth, randomize_features)
        super(CartSymSplitter, self).__init__(fact_neg, fact_pos, depth)

    def fit(self, X, y):
        """
            Fits to the data.

            :param fact_neg:    Weight on negative local density.
            :param fact_pos:    Weight on positive local density.
            :param X:           Position of each point.
            :param y:           Binary class of the point, in {0,1}.

            :return:            None.
        """
        # Run TreeRank to obtain a partition of the space, with a full tree.
        self._treerank.fit_pairs(X, y)
        self._find_split_from_fitted_tree()

    def plot(self, region, *argv, **kwargs):
        """Plots the region delimited by the splitter."""
        raise Exception("TODO: Not yet implemented.")

    def cut_region(self, region):
        """Return the regions in and out when the splits cuts the region."""
        raise Exception("TODO: Not yet implemented.")

    # --------------- Private methods ---------------

    def _define_treerank(self, depth, randomize_features):
        # One can also use CrossSymSplitter here.
        if randomize_features is None:
            split_inst = DiagSymSplitter
            # CrossSymSplitter
        else:
            split_inst = lambda f_n, f_p: RandomSymFeaturesSplitter(
                DiagSymSplitter(f_n, f_p), randomize_features)
            # CrossSymSplitter
        self._treerank = SimTreeRank(split_inst, depth,
                                     change_emp_measure=False)
