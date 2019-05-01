"""
    Implementation of pruned treerank

    Now BreimanPruning takes the averages of the lambdas
    instead of best lambda over the average AUC.

    TODO: Implement the structural AUC maximization.
"""
from operator import itemgetter
import numpy as np
from sklearn.model_selection import KFold

from treerank.utils import auc


def one_iteration(old_lambda, srt_term_nodes):
    """Finds the smallest delta AUC and updates lambda."""
    # print("Input " , old_lambda, srt_term_nodes)
    new_lambda = max(srt_term_nodes[0][0], old_lambda)
    srt_term_nodes[0][1].weight = 0
    # The parent of the terminal node is not always terminal,
    # hence the check.
    if srt_term_nodes[0][1].parent.is_terminal_node():
        # Insert the parent into the sorted list:
        new_delta_auc = srt_term_nodes[0][1].delta_auc()
        new_term_node = srt_term_nodes[0][1].parent
        ind_new_node = next((i+1 for i, t in enumerate(srt_term_nodes[1:])
                             if t[0]), len(srt_term_nodes))
        return new_lambda, srt_term_nodes[1:ind_new_node] \
                + [(new_delta_auc, new_term_node)] \
                + srt_term_nodes[ind_new_node:]
    return new_lambda, srt_term_nodes[1:]

def prune(model, lamb):
    """Iterates through the terminal nodes."""
    srt_term_nodes = sorted([(s.delta_auc(), s)
                             for s in model.get_terminal_nodes()],
                            key=itemgetter(0))
    cur_lambda = 0
    while cur_lambda < lamb:
        cur_lambda, srt_term_nodes = one_iteration(
            cur_lambda, srt_term_nodes)

class BreimanPruning:
    """Introduces the simple pruning mechanism of [1]."""

    def __init__(self, n_folds, model_instantiator):
        self.model = model_instantiator()
        self.n_folds = n_folds
        # private
        self.__model_instantiator = model_instantiator

    def get_roc_curve(self):
        """Returns the ROC curve of the model on training data."""
        return self.model.get_roc_curve()

    def score(self, X):
        """Returns the score of the model on X."""
        return self.model.score(X)

    def fit(self, X, y):
        """Fits the m"""
        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        opt_lambdas = [self.__get_best_lambda_cv(X, y, train_index, test_index)
                       for train_index, test_index in kfold.split(X)]

        chosen_lambda = np.mean(opt_lambdas)
        self.model.fit(X, y)
        prune(self.model, chosen_lambda)

    def __get_best_lambda_cv(self, X, y, train_index, test_index):
        """Finds the lambda that gives the best AUC on test_indexes."""
        cur_lambda = 0
        cv_auc_per_lambda = list()

        # Fits the model.
        model = self.__model_instantiator()
        model.fit(X[train_index], y[train_index])
        term_nodes = sorted([(s.delta_auc(), s)
                             for s in model.get_terminal_nodes()],
                            key=itemgetter(0))

        while term_nodes[0][1].parent is not None:
            auc_test = auc(model.score(X[test_index]), y[test_index])
            cv_auc_per_lambda.append((cur_lambda, auc_test))
            cur_lambda, term_nodes = one_iteration(cur_lambda, term_nodes)

        return max(cv_auc_per_lambda, key=itemgetter(1))[0]
