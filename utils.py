"""
    Utilities for the experiments.
"""
import os
import sys
import logging
import math
import json
from datetime import datetime
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# Model imports
from metric_learn.lmnn import LMNN
from treerank.nn_model import NeuralModel
from treerank.pickable_rf import sym_ranking_forest_parallel

from treerank.utils import (
    det_curve,
    roc_curve,
    get_random_pairs
)
from treerank.other_utils import save_model

from data_utils import load_preprocess_data


with open("treerank/params.json", "rt") as f:
    PARAMS = json.load(f)

class ModelInterface:
    """
        Interface to several models:
            * LMNN
            * angular-margin
            * RF
    """
    #pylint: disable-msg=missing-docstring

    def __init__(self, model_name):
        self.ul_model = None # Underlying model
        self.name = model_name
        # Linear models
        if model_name in {"LMNN"}:
            self.model_type = "linear"
            self.__init_linear_model(model_name)

        # NN-based models
        if model_name in {"nn_sce"}:
            self.model_type = "NN"
            self.__init_neural_model(model_name)

        # TreeRank models
        if model_name in {"RF"}:
            self.model_type = "treerank"
            self.__init_treerank_model(model_name)

    def fit(self, X, y):
        if self.model_type == "linear":
            filt_classes = np.array([key for key, val in Counter(y).items()
                                     if val <= 3])
            filt = np.array([cl not in filt_classes for cl in y])
            X = X[filt]
            y = y[filt]
        self.ul_model.fit(X, y)

    def score(self, X, batchsize=10**4):
        n = X.shape[0]
        scores = list()
        i = 0
        while i < n:
            print("Processed {} instances out of {} - ".format(i, n)
                  + datetime.now().ctime())
            scores.append(self.__score_batch(X[i:min(i+batchsize, n)]))
            i += batchsize
        return np.hstack(scores)

    # ---------- Private methods

    def __init_linear_model(self, model_name):
        if model_name == "LMNN":
            self.ul_model = LMNN(regularization=0.01) # use_pca=True

    def __init_neural_model(self, model_name):
        if model_name == "nn_sce":
            self.ul_model = NeuralModel()

    def __init_treerank_model(self, model_name):
        if model_name == "RF":
            self.ul_model = sym_ranking_forest_parallel()

    def __score_batch(self, X):
        if self.model_type == "linear":
            d = X.shape[1]//2
            transformed_x1 = self.ul_model.transform(X[:, :d])
            transformed_x2 = self.ul_model.transform(X[:, d:])
            return (transformed_x1*transformed_x2).sum(axis=1)
        if self.model_type == "NN":
            d = X.shape[1]//2
            encode_left = self.ul_model.encode(X[:, :d])
            encode_right = self.ul_model.encode(X[:, d:])
            norm_left = np.linalg.norm(encode_left, axis=1)
            norm_right = np.linalg.norm(encode_right, axis=1)
            return (encode_right*encode_left).sum(axis=1)/(
                norm_left*norm_right)
        # self.model_type == "treerank":
        return self.ul_model.score(X)


def fit_n_save_roc(unfitted_model, dbname, model_folder="models",
                   roc_folder="tmp", n_test_pairs=PARAMS["n_test_pairs"]):
    """Fits the model and saves the ROC scores."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                        format='%(levelname)s - %(asctime)s - %(message)s',
                        datefmt='%d/%m %I:%M:%S')
    # logging.basicConfig(level=logging.DEBUG,
    #                     filename='experiments.log',
    #                     format='%(asctime)s - %(message)s')
    print("## Loading data... - " + datetime.now().ctime())
    X_train, y_train, X_test, y_test = load_preprocess_data(dbname=dbname)
    logging.debug("Dimensionality of X_train: %s x %s",
                  X_train.shape[0], X_train.shape[1])
    logging.debug("Dimensionality of X_test: %s x %s",
                  X_test.shape[0], X_test.shape[1])

    print("### Fitting the model... - " + datetime.now().ctime())
    model = unfitted_model
    model.fit(X_train, y_train)

    print("### Preparing paths... - " + datetime.now().ctime())
    # Making output folders
    if not os.path.exists("{}/".format(model_folder)):
        os.makedirs("{}/".format(model_folder))

    if not os.path.exists("{}/".format(roc_folder)):
        os.makedirs("{}/".format(roc_folder))

    # Finding right path names
    cur_i = -1
    model_filename, score_filename, z_filename = ["."]*3
    while cur_i < 20:
        if not(os.path.exists(model_filename)
               or os.path.exists(score_filename)
               or os.path.exists(z_filename)):
            print("### Saving as model {}... - ".format(cur_i)
                  + datetime.now().ctime())
            break
        cur_i += 1
        model_filename = "{}/model_{}_{}_{}".format(model_folder, dbname,
                                                    model.name, cur_i)
        score_filename = "{}/scores_{}_{}_{}.npy".format(roc_folder, dbname,
                                                         model.name, cur_i)
        z_filename = "{}/z_{}_{}_{}.npy".format(roc_folder, dbname,
                                                model.name, cur_i)

    print("### Saving the model... - " + datetime.now().ctime())
    # Will work only for certain types of models.
    save_model(model_filename, model.ul_model)

    print("### Saving the ROC curve... - " + datetime.now().ctime())
    X_test_pairs, z_test_pairs = get_random_pairs(X_test, y_test, n_test_pairs)
    s_test = model.score(X_test_pairs)
    s_test.dump(score_filename)
    z_test_pairs.dump(z_filename)

def plot_rocs(dbname, model_names, infolder="", logscale=True,
              filtered={"RF_6": "Ranking Forest",
                        "LMNN_1": "LMNN",
                        "nn_sce_0": "Neural Network"}):
    """Plots the saved ROCs."""
    for model_name in model_names:
        cur_i = -1
        s_filename = "."
        z_filename = "."
        while cur_i < 20:
            if (cur_i >= 0 and os.path.exists(z_filename)
                    and os.path.exists(s_filename)):
                if (filtered is None) or ("{}_{}".format(model_name, cur_i) in
                                          filtered.keys()):
                    assert (os.path.exists(s_filename) and
                            os.path.exists(z_filename))
                    s_test = np.load(s_filename)
                    z_test_pairs = np.load(z_filename)
                    if filtered is None:
                        cur_label = "{}_{}".format(model_name, cur_i)
                    else:
                        cur_label = filtered["{}_{}".format(model_name, cur_i)]
                    plt.plot(*roc_curve(s_test, z_test_pairs), # det_curve
                             label=cur_label)
            cur_i += 1
            s_filename = "{}/scores_{}_{}_{}.npy".format(infolder, dbname,
                                                         model_name, cur_i)
            z_filename = "{}/z_{}_{}_{}.npy".format(infolder, dbname,
                                                    model_name, cur_i)

    # plt.plot([0, 0.01, 1], [1, 0.99, 0], color="black", label="random") #DET
    plt.plot([0, 0.01, 1], [0, 0.01, 1], color="black", label="Random") #ROC

    if logscale:
        plt.xscale("log")
        plt.xlim([0.00001, 1.])
    plt.xlabel("FPR") # FAR")
    plt.ylabel("TPR")  # FRR")
    plt.grid()
    plt.title("{} ROC".format(dbname))
    plt.tight_layout()
    plt.legend(loc="lower right")

def plot_rocs_n_save(outfolder, dbname, *argv, **kwargs):
    """Plots the saved ROCS and saves them."""
    figsize = (4, 4)
    plt.figure(figsize=figsize)
    plot_rocs(dbname, *argv, **kwargs)

    if not os.path.exists("{}/".format(outfolder)):
        os.makedirs("{}/".format(outfolder))
    plt.savefig("{}/roc_{}.pdf".format(outfolder, dbname), format="pdf")
