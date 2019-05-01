"""
    Models stored here for convenience.
"""
# Main models imports
from treerank.treerank import TreeRank
from treerank.treerank_bagging import BaggedModel
from treerank.treerank_similarity import (
    SimTreeRank,
    SimBagging,
)

from treerank.treerank_pruning import BreimanPruning

# Splitters imports
## Non-symmetric splitters
from treerank.cart_splitter import CartSplitter
from treerank.split_space.random_features_splitter \
        import RandomFeaturesSplitter

## Symmetric splitters
from treerank.cart_sym_splitter import CartSymSplitter
from treerank.split_space.diag_sym_splitter import DiagSymSplitter
from treerank.split_space.cross_sym_splitter import CrossSymSplitter
from treerank.split_space.random_sym_features_splitter \
        import RandomSymFeaturesSplitter

#  Bipartite ranking models

## Regular trees

## Trees with pruning

## Ranking forests
def ranking_forest_cart_splitter(params):
    """Returns a bagged treerank with cart splitter."""
    expected_params = {"depth_splitter", "rand_feat_splitter",
                       "depth_treerank", "rand_feat_treerank",
                       "n_sample", "card_sample"}
    assert expected_params.issubset(set(params.keys()))

    def split_inst(f_n, f_p):
        return  RandomFeaturesSplitter(
            CartSplitter(f_n, f_p, params["depth_splitter"],
                         params["rand_feat_splitter"]),
            params["rand_feat_treerank"])

    def model_inst():
        return TreeRank(split_inst, params["depth_treerank"])

    return BaggedModel(params["n_sample"], params["card_sample"], model_inst)

# ----------------------------------------
#  Similarity ranking models

## Regular trees
def sym_treerank_cross_splitter(n_pairs, depth):
    """Similarity TreeRank with cross split."""
    return SimTreeRank(n_pairs, CrossSymSplitter, depth, True)

def sym_treerank_diagsym_splitter(n_pairs, depth):
    """Similarity TreeRank with diag sym split."""
    return SimTreeRank(n_pairs, DiagSymSplitter, depth, True)

## Trees with pruning

def sym_pruned_treerank_diagsym_splitter(params):
    """Pruned symmetric treerank with diagonal splitter."""
    expected_params = {"depth_splitter", "depth_treerank",
                       "card_sample", "n_folds"}
    assert expected_params.issubset(set(params.keys()))

    def split_inst(f_n, f_p):
        return CartSymSplitter(f_n, f_p,
                               depth=params["depth_splitter"])

    def model_inst():
        return SimTreeRank(split_inst,
                           params["depth_treerank"],
                           params["card_sample"])

    return BreimanPruning(params["n_folds"], model_inst)

## Ranking forests
def sym_ranking_forest_cross_splitter(params):
    """Returns a bagged treerank with cross_splitter."""
    expected_params = {"depth_treerank", "rand_feat_treerank",
                       "n_sample", "card_sample"}
    assert expected_params.issubset(set(params.keys()))

    def split_inst(f_n, f_p):
        return RandomSymFeaturesSplitter(
            CrossSymSplitter(f_n, f_p),
            params["rand_feat_treerank"])

    def model_inst():
        return TreeRank(split_inst, params["depth_treerank"])

    return SimBagging(params["n_sample"], params["card_sample"], model_inst)

def sym_ranking_forest_diagsym_splitter(params):
    """Returns a bagged treerank with a diag sym splitter."""
    expected_params = {"depth_treerank", "rand_feat_treerank",
                       "n_sample", "card_sample"}
    assert expected_params.issubset(set(params.keys()))

    def split_inst(f_n, f_p):
        return RandomSymFeaturesSplitter(
            CrossSymSplitter(f_n, f_p),
            params["rand_feat_treerank"])

    def model_inst():
        return SimTreeRank(split_inst, params["depth_treerank"])

    return SimBagging(params["n_sample"], params["card_sample"], model_inst)

def sym_ranking_forest_cart_splitter(params):
    """Returns a bagged treerank with a sym cart."""
    expected_params = {"depth_splitter", "rand_feat_splitter",
                       "depth_treerank", "rand_feat_treerank",
                       "n_sample", "card_sample"}
    assert expected_params.issubset(set(params.keys()))

    def split_inst(f_n, f_p):
        return RandomSymFeaturesSplitter(
            CartSymSplitter(f_n, f_p,
                            params["depth_splitter"],
                            params["rand_feat_splitter"]),
            params["rand_feat_treerank"])

    def model_inst():
        return SimTreeRank(split_inst, params["depth_treerank"])

    return SimBagging(params["n_sample"], params["card_sample"], model_inst)

def sym_ranking_forest_cart_splitter_left_ROC(params):
    """Returns a bagged treerank with a sym cart."""
    expected_params = {"depth_splitter", "rand_feat_splitter",
                       "depth_treerank", "rand_feat_treerank",
                       "n_sample", "card_sample",
                       "alpha_cons"}
    assert expected_params.issubset(set(params.keys()))

    def split_inst(f_n, f_p):
        return RandomSymFeaturesSplitter(
            CartSymSplitter(f_n, f_p,
                            params["depth_splitter"],
                            params["rand_feat_splitter"]),
            params["rand_feat_treerank"])

    def model_inst():
        return SimTreeRank(split_inst, params["depth_treerank"],
                           alpha_constraint=params["alpha_cons"])

    return SimBagging(params["n_sample"], params["card_sample"], model_inst)
