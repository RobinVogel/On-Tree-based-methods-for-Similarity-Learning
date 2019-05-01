"""
    Parallel implementation of the ranking forest for similarity
    learning.
"""
import json
from treerank.cart_sym_splitter import CartSymSplitter
from treerank.split_space.random_sym_features_splitter \
        import RandomSymFeaturesSplitter

from treerank.treerank_similarity import (
    SimTreeRank,
    SimBagging,
)

with open("treerank/params.json", "rt") as f:
    PARAMS = json.load(f)

DEPTH_SPLITTER = PARAMS["depth_splitter"]
RAND_FEAT_SPLITTER = PARAMS["rand_feat_splitter"]
RAND_FEAT_TREERANK = PARAMS["rand_feat_treerank"]
DEPTH_TREERANK = PARAMS["depth_treerank"]
ALPHA_CONS = PARAMS["alpha_cons"]
N_SAMPLE = PARAMS["n_sample"]
CARD_SAMPLE = PARAMS["card_sample"]

def split_inst(f_n, f_p):
    """Instantiator of a splitter."""
    return RandomSymFeaturesSplitter(
        CartSymSplitter(f_n, f_p,
                        DEPTH_SPLITTER,
                        RAND_FEAT_SPLITTER),
        RAND_FEAT_TREERANK)

def sim_parallel_inst():
    """Instantiator for the Sym parallel, pickable."""
    return SimTreeRank(split_inst, DEPTH_TREERANK, alpha_constraint=ALPHA_CONS)

def get_fitted_model(model_instantiator, X_resampled, y_resampled):
    """Returns a fitted model for pairwise problems."""
    cur_model = model_instantiator()
    cur_model.fit_pairs(X_resampled, y_resampled)
    return cur_model

class PickableRandomForest(SimBagging):
    """Bagging for similarity treerank."""
    def __init__(self, n_sample, card_sample):
        super(PickableRandomForest, self).__init__(n_sample, card_sample,
                                                   sim_parallel_inst)
        self.parallel = True

def sym_ranking_forest_parallel():
    """"The parallel aspect gives us constraints to handle the init."""
    return PickableRandomForest(N_SAMPLE, CARD_SAMPLE)
