"""
    Coordinate Splitter.
"""
import logging
import numpy as np
from .space_splitter import SpaceSplitter

class CoordinateSplitter(SpaceSplitter):
    """Separates the space on one coordinate."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, fact_neg, fact_pos):
        super(CoordinateSplitter, self).__init__(fact_neg, fact_pos)
        # Protected
        self._only_upper = False
        # Private
        self.__cut_dim = None
        self.__cut_val = None
        self.__lower_than = None
        self.__eps = np.finfo(float).eps
        self.__only_lower = False

    def fit(self, X, y):
        fit_params = self._space_fit(X, y)
        self.best_value = fit_params["best_value"]
        self.__cut_dim = fit_params["cut_dim"]
        self.__cut_val = fit_params["cut_val"]
        self.__lower_than = fit_params["lower_than"]
        self.n_pos = fit_params["n_pos"]
        self.n_neg = fit_params["n_neg"]

    def check(self, X):
        return (X[:, self.__cut_dim] <= self.__cut_val) == self.__lower_than

    def plot(self, region, *argv, **kwargs):
        region.plot_coord_split(self.__cut_val, self.__cut_dim,
                                *argv, **kwargs)

    def cut_region(self, region):
        splits = region.split_coord(self.__cut_dim, self.__cut_val)
        if self.__lower_than:
            return splits[1], splits[0]
        return splits

    # --------------- Protected methods ---------------

    def _space_fit(self, X, y):
        """
            Execute line_fit on all of the dimensions of X to find
            the best split for the entropic measure presented in [1],
            page 41. Assumes y in {0,1}.

            Refer to the docstring of line_fit for more details.
        """
        dims = X.shape[1]
        best_split = {"best_value": -float("inf")}
        logging.debug("Call space fit")
        for d in range(dims):
            x = X[:, d]
            cur_split = self._line_fit(x, y)
            if cur_split["best_value"] > best_split["best_value"]:
                best_split = cur_split
                best_split["cut_dim"] = d
        logging.debug("Call space fit - dim %s cut_val %.2f lower %s",
                      best_split["cut_dim"], best_split["cut_val"],
                      best_split["lower_than"])
        return best_split

    def _line_fit(self, x, y):
        r"""
            Fits a line to optimize for the entropic measure.

            Given a weight for the positive instances and negative
            instances, partitions the space of real values as
            [-\infty, A] or [A, +\infty] to maximize the entropic
            measure presented in [1], page 41.

            :return:    A dictionary w/ keys: best_value, cut_val,
                        n_neg, n_pos, lower_than.

            lower_than is True if the partition is [-\infty, A],
            and False otherwise. See docstring of unoriented_line_fit
            for a description of others keys.
        """
        ind_sort = np.argsort(x)
        y_sorted = y[ind_sort]
        x_sorted = x[ind_sort]
        x_unique = np.unique(x_sorted)

        assert not(self.__only_lower and self.__only_lower)
        # logging.debug("Call line fit")
        # We keep the lower split:
        if self.__only_lower:
            best_split_left = self.__unoriented_line_fit(
                y_sorted, x_sorted, x_unique)
            return dict(list(best_split_left.items())
                        + [("lower_than", True)])

        # We keep the upper split:
        if self._only_upper:
            best_split_right = self.__unoriented_line_fit(
                np.flip(y_sorted, axis=0), np.flip(x_sorted, axis=0),
                x_unique[::-1])
            return dict(list(best_split_right.items())
                        + [("lower_than", False)])

        # We keep the best split of both sides:
        best_split_left = self.__unoriented_line_fit(
            y_sorted, x_sorted, x_unique)
        best_split_right = self.__unoriented_line_fit(
            np.flip(y_sorted, axis=0), np.flip(x_sorted, axis=0),
            x_unique[::-1])

        if best_split_left["best_value"] >= best_split_right["best_value"]:
            return dict(list(best_split_left.items()) + [("lower_than", True)])
        return dict(list(best_split_right.items()) + [("lower_than", False)])

    # --------------- Private methods ---------------

    def __unoriented_line_fit(self, y_sorted, x_sorted, x_vals):
        r"""
            Finds a partition of the type [-\infty, A] that contains
            the elements of x_sorted and maximizes for
            factors[1]*n_pos - factors[2]*n_neg where n_pos and n_neg
            are the numbers of positives (y=1) and negatives (y=0)
            in partition.

            :param y_sorted:    Array of the class values, sorted
                                according to x_sorted. In {0,1}
            :param x_sorted:    Array of feature values in R.
            :param x_vals:      Unique values of x_sorted.

            :return:            A dictionary that contains: best_value,
                                cut_val, n_neg, n_pos.

            The value ext_margin can be negative if x_sorted is descending.
            best_value is the best sum of weights.
            cut_val is the middle of two points of x_sorted or extremes
            points +/- eps.
            n_pos, n_neg are the number of positives and negatives in
            the partition, from y_sorted.
        """
        best_split = {"best_value": 0., "cut_val": x_sorted[0] - self.__eps,
                      "n_neg": 0, "n_pos": 0}
        n_tot = y_sorted.shape[0]
        i_s = 0
        cur_best_val = 0
        # Stores cardinal of neg, pos in split.
        cur_card = np.array([0, 0])
        i_val = 0
        n_vals = len(x_vals)
        while i_val < n_vals:
            delta_card = np.array([0, 0])
            while i_s <= n_tot-1 and x_sorted[i_s] == x_vals[i_val]:
                delta_card += np.array([1-y_sorted[i_s], y_sorted[i_s]])
                i_s += 1
            cur_card += delta_card
            cur_best_val = cur_best_val \
                + (np.array([self.fact_neg, self.fact_pos])*delta_card).sum()
            # logging.debug("cur_best_val %.2f", cur_best_val)
            if cur_best_val > best_split["best_value"]:
                if i_val == n_vals - 1:
                    cut_val = x_vals[i_val] + self.__eps
                else:
                    cut_val = (x_vals[i_val] + x_vals[i_val+1])/2
                best_split = {"best_value": cur_best_val, "cut_val": cut_val,
                              "n_neg": cur_card[0], "n_pos": cur_card[1]}
            i_val += 1
        return best_split
