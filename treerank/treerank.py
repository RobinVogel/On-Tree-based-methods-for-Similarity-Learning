"""
    TreeRank algorithm
"""
import logging
import numpy as np
from treerank.binary_tree import BinaryTree

class TreeRank:
    """
        Implements the TreeRank algorithm.

        See [1], page 41.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, splitter_instantiator, depth,
                 change_emp_measure=True, alpha_constraint=None):
        self.depth = depth
        self.tree = BinaryTree()
        self.splitter_instantiator = splitter_instantiator
        self.change_empirical_measure = change_emp_measure
        self.fact_neg = None
        self.fact_pos = None
        # Defines a constraint on alpha that restricts the fitting:
        self.alpha_cons = alpha_constraint
        # private
        self.__n_pos = None
        self.__n_neg = None

    def set_fixed_factors(self, fact_neg, fact_pos):
        """To be called before fitting when using fixed factors."""
        self.fact_neg = fact_neg
        self.fact_pos = fact_pos

    def get_roc_curve(self):
        """Returns the ROC curve on training data."""
        def rec(tree):
            part_roc = list()
            if tree.weight > 0:
                if tree.left is not None:
                    part_roc = part_roc + rec(tree.left)
                if tree.data:
                    part_roc.append((tree.data["alpha_mid"],
                                     tree.data["beta_mid"]))
                if tree.right is not None:
                    part_roc = part_roc + rec(tree.right)
            return part_roc
        return zip(*([(0., 0.)] + rec(self.tree) + [(1., 1.)]))

    def get_terminal_nodes(self):
        """Returns the terminal nodes of the tree."""
        def rec(tree):
            if tree.is_terminal_node():
                return [tree]
            descendant_leaves = list()
            if tree.left.data and tree.left.weight > 0:
                descendant_leaves += rec(tree.left)
            if tree.right.data and tree.right.weight > 0:
                descendant_leaves += rec(tree.right)
            return descendant_leaves
        return rec(self.tree)

    def get_leaf_partitions(self):
        """
            Returns the list of partitions of the tree and # pos and neg.

            Returns the list of tuples (partition_id, (n_neg, n_pos))
            from the tree class, once fitted.
        """
        depth = self.depth
        def rec(tree, d, n_neg, n_pos):
            # If the tree is a leaf, return 0 and cardinals:
            if not(tree.data) or tree.weight == 0:
                return [(0, (n_neg, n_pos))]
            # Otherwise:
            left_side_parts = [((2**(depth-d-1) + i_part), cards)
                               for i_part, cards in rec(
                                   tree.left, d+1,
                                   tree.splitter.n_neg, tree.splitter.n_pos)]
            right_side_parts = [(i_part, cards)
                                for i_part, cards in rec(
                                    tree.right, d+1,
                                    n_neg - tree.splitter.n_neg,
                                    n_pos - tree.splitter.n_pos)]

            return left_side_parts + right_side_parts
        return rec(self.tree, 0, self.__n_neg, self.__n_pos)

    def get_rank_cell(self, X):
        """Returns the rank of the X's."""
        def rec(X, tree):
            """Returns (rank value of the X's, # places occupied above)."""
            ranks = np.zeros(X.shape[0])
            if not(tree.data) or tree.weight == 0:
                return np.zeros(X.shape[0]), 1
            if X.shape[0] == 0:
                # If there are no elements in the cell
                return np.zeros(X.shape[0]), rec(X, tree.left)[1]
            in_region = tree.splitter.check(X)
            ranks[in_region], n_ranks_above = rec(X[in_region], tree.left)
            ranks[~in_region], _ = rec(X[~in_region], tree.right)
            ranks[~in_region] += n_ranks_above
            return ranks, _
        return rec(X, self.tree)[0]

    def score(self, X):
        """Computes a score between 0 and 1 for the elements in X."""
        def rec(tree, X):
            """Returns the score of X"""
            if X.shape[0] == 0 or not(tree.data) or tree.weight == 0:
                return np.zeros(X.shape[0])
            in_region = tree.splitter.check(X)
            scores = in_region.astype(float)
            scores[in_region] += 0.5*rec(tree.left, X[in_region])
            scores[~in_region] += 0.5*rec(tree.right, X[~in_region])
            return scores
        return 0.5*rec(self.tree, X)

    def fit(self, X, y, spec_left_fit=False):
        """Fits the tree."""
        self.__n_pos = y.sum()
        self.__n_neg = len(y) - self.__n_pos
        logging.debug("Start fitting of the tree.")
        self.__recursive_fit(X, y, self.tree,
                             {"depth": 0, "alpha_min": 0, "alpha_max": 1,
                              "beta_min": 0, "beta_max": 1},
                             spec_left_fit=spec_left_fit)

    def __instantiate_splitter(self, p_node):
        """Defines the right splitter given we chose fixed weights or not."""
        if self.change_empirical_measure:
            return self.splitter_instantiator(
                -(p_node["beta_max"] - p_node["beta_min"])/self.__n_neg,
                (p_node["alpha_max"] - p_node["alpha_min"])/self.__n_pos)

        assert self.fact_neg is not None or self.fact_pos is not None
        return self.splitter_instantiator(self.fact_neg, self.fact_pos)

    def __recursive_fit(self, X, y, tree, p_node, spec_left_fit=False):
        """Recursive fitting of the tree."""
        #pylint: disable-msg=too-many-arguments
        assert p_node["depth"] <= self.depth
        logging.debug("Depth %s - alphas (%.2f, %.2f) betas (%.2f, %.2f)",
                      p_node["depth"],
                      p_node["alpha_min"], p_node["alpha_max"],
                      p_node["beta_min"], p_node["beta_max"])
        n_pos = y.sum()
        n_neg = y.shape[0] - n_pos
        if p_node["depth"] == self.depth:
            logging.debug("Depth %s - Leaf - maximum depth reached.",
                          p_node["depth"])
            tree.data = None
        elif n_pos in (0, y.shape[0]):
            logging.debug("Depth %s - Leaf - pure node",
                          p_node["depth"])
            tree.data = None
        else:
            tree.splitter = self.__instantiate_splitter(p_node)
            if spec_left_fit:
                tree.splitter.fit(X, y, only_lower=True)
            else:
                tree.splitter.fit(X, y)
            logging.debug("Depth %s - Leaf - Done fitting the leaf.",
                          p_node["depth"])

            all_data = (tree.splitter.n_pos == n_pos) and \
                    (tree.splitter.n_neg == n_neg)
            no_data = (tree.splitter.n_pos == 0) and (tree.splitter.n_neg == 0)

            if all_data or no_data:
                logging.debug("Depth %s - Leaf - Split did not improve.",
                              p_node["depth"])
                tree.data = None
                tree.splitter = None
            else:
                tree.right = BinaryTree(parent=tree)
                tree.left = BinaryTree(tree)
                self.__good_split(X, y, tree, p_node,
                                  spec_left_fit=spec_left_fit)

    def __good_split(self, X, y, tree, p_node, spec_left_fit=False):
        """
            Computes the properties of node tree and runs fit on children.

            Calls __recursive_fit on tree.right and tree.left after updating
            the tree.data value.
        """
        #pylint: disable-msg=too-many-arguments
        # Compute the alphas and betas for the child nodes.
        beta_mid = p_node["beta_min"] + tree.splitter.n_pos/self.__n_pos
        alpha_mid = p_node["alpha_min"] + tree.splitter.n_neg/self.__n_neg

        logging.debug("Depth %s - Cut: alpha %.2f beta %.2f",
                      p_node["depth"], alpha_mid, beta_mid)

        tree.data = dict(**{"alpha_mid": alpha_mid, "beta_mid": beta_mid},
                         **p_node)

        # The left node optimizes for the left side of the ROC curve,
        # and the right node optimizes for the right side.
        filt = tree.splitter.check(X)

        logging.debug("Depth %s - Left node.", p_node["depth"])
        p_node_left = {"depth": p_node["depth"] + 1,
                       "alpha_min": p_node["alpha_min"],
                       "alpha_max": alpha_mid,
                       "beta_min": p_node["beta_min"],
                       "beta_max": beta_mid}
        self.__recursive_fit(X[filt], y[filt], tree.left, p_node_left,
                             spec_left_fit=spec_left_fit)

        if (self.alpha_cons is None) or (self.alpha_cons > alpha_mid):
            logging.debug("Depth %s - Right node.", p_node["depth"])
            p_node_right = {"depth": p_node["depth"] + 1,
                            "alpha_min": alpha_mid,
                            "alpha_max": p_node["alpha_max"],
                            "beta_min": beta_mid,
                            "beta_max": p_node["beta_max"]}
            self.__recursive_fit(X[~filt], y[~filt], tree.right, p_node_right,
                                 spec_left_fit=False)
