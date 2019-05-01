"""
    Cross symmetric splitter.
"""
import numpy as np

from .space_splitter import SpaceSplitter

class CrossSymSplitter(SpaceSplitter):
    """Coordinate space splitter."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, fact_neg, fact_pos):
        super(CrossSymSplitter, self).__init__(fact_neg, fact_pos)
        # Do we impose the diagonal to be in the set ?
        self.__diagonal_in = False
        self.__cut_dim = None
        self.__cut_val = None
        self.__along_diag = None
        self.__eps = np.finfo(float).eps
        self.__dim = None

    def fit(self, X, y):
        assert len(X.shape) > 1 and X.shape[1] % 2 == 0
        self.__dim = X.shape[1]
        fit_params = self.__space_fit(X, y)
        self.best_value = fit_params["best_value"]
        self.__cut_dim = fit_params["cut_dim"]
        self.__cut_val = fit_params["cut_val"]
        self.__along_diag = fit_params["along_diag"]
        self.n_pos = fit_params["n_pos"]
        self.n_neg = fit_params["n_neg"]

    def check(self, X):
        d = self.__cut_dim
        X_dim = X[:, [d, d+self.__dim//2]]
        X_dim_mins = np.min(X_dim, axis=1)
        X_dim_maxs = np.max(X_dim, axis=1)
        in_align_diag = np.logical_or(X_dim_mins > self.__cut_val,
                                      X_dim_maxs < self.__cut_val)
        return in_align_diag == self.__along_diag

    def plot(self, region, *argv, **kwargs):
        """Plots the region delimited by the splitter."""
        region.cross_plot(self.__cut_val, *argv, **kwargs)

    def cut_region(self, region):
        """Return the region cut by the split."""
        splits = region.cross_split(self.__cut_val)
        if self.__along_diag:
            return splits
        return splits[1], splits[0]

    # --------------- Private methods ---------------

    def __space_fit(self, X, z):
        best_split = {"best_value": -float("inf")}
        for d in range(self.__dim//2):
            X_dim = X[:, [d, d+self.__dim//2]]
            cur_split = self.__line_fit(X_dim, z)
            if cur_split["best_value"] > best_split["best_value"]:
                best_split = cur_split
                best_split["cut_dim"] = d
        return best_split

    def __line_fit(self, X_dim, z):
        #pylint: disable-msg=too-many-locals
        X_dim_mins = np.min(X_dim, axis=1)
        X_dim_maxs = np.max(X_dim, axis=1)
        argsort_min = np.argsort(X_dim_mins)
        argsort_max = np.argsort(X_dim_maxs)
        X_dim_mins = X_dim_mins[argsort_min]
        X_dim_maxs = X_dim_maxs[argsort_max]
        z_sorted_min = z[argsort_min]
        z_sorted_max = z[argsort_max]
        n_tot = z.shape[0]
        n_pos = np.sum(z)
        n_neg = n_tot - n_pos

        def __get_best(cur_best_split, cur_card):
            cur_best_val = (np.array([self.fact_neg,
                                      self.fact_pos])*cur_card).sum()
            if cur_best_split["best_value"] < cur_best_val:
                return {"best_value": cur_best_val, "n_neg": cur_card[0],
                        "n_pos": cur_card[1], "along_diag": True}, True
            if not self.__diagonal_in:
                cur_best_val = (np.array([self.fact_neg, self.fact_pos])*(
                    np.array([n_neg, n_pos]) - cur_card)).sum()
                if cur_best_split["best_value"] < cur_best_val:
                    return {"best_value": cur_best_val,
                            "n_neg": n_neg - cur_card[0],
                            "n_pos": n_pos - cur_card[1],
                            "along_diag": False}, True
            return cur_best_split, False

        best_split = {"best_value": -float("inf")}
        cur_card = np.array([n_neg, n_pos])
        x_vals = np.unique(np.hstack([X_dim_mins, X_dim_maxs]))
        best_split, _ = __get_best(best_split, cur_card)
        best_split["cut_val"] = np.min(X_dim_mins) - self.__eps

        # In the beginning of the loop, all points are inside the region,
        # then when I pass some mins they are out, when I pass some maxs
        # they are back in. At the end everyone is inside.
        i_min, i_max = 0, 0
        n_vals = len(x_vals)
        i_val = 0
        while i_val < n_vals:
            delta_card = np.array([0, 0])
            while i_min <= n_tot - 1 and X_dim_mins[i_min] == x_vals[i_val]:
                delta_card -= np.array([1-z_sorted_min[i_min],
                                        z_sorted_min[i_min]])
                i_min += 1
            while i_max <= n_tot - 1 and X_dim_maxs[i_max] == x_vals[i_val]:
                delta_card += np.array([1-z_sorted_max[i_max],
                                        z_sorted_max[i_max]])
                i_max += 1
            cur_card += delta_card
            best_split, better_split = __get_best(best_split, cur_card)
            if better_split:
                if i_val == n_vals - 1:
                    best_split["cut_val"] = x_vals[i_val] + self.__eps
                else:
                    best_split["cut_val"] = (x_vals[i_val] + x_vals[i_val+1])/2
            i_val += 1
        return best_split
