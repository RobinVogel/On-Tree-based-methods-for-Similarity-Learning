"""
    Diadic partitioning.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from .space_splitter import SpaceSplitter

class DiadicPartition(SpaceSplitter):
    """Splits the space in a diadic manner."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, fact_neg, fact_pos, depth):
        super(DiadicPartition, self).__init__(fact_neg, fact_pos)
        self.__depth = depth
        self.__eps = np.finfo(float).eps
        self.__x_mins, self.__x_maxes = [np.array([])]*2
        self._l_parts, self._r_parts = [list()]*2

    def fit(self, X, y):
        """Fits the partition to maximize an entropic measure, see [1]."""
        partitions = self.__get_uniform_diadic_partition(X, y)
        self._maximize_crit_from_parts(partitions)
        for _, (p_n_neg, p_n_pos) in self._l_parts:
            self.n_neg += p_n_neg
            self.n_pos += p_n_pos

    def check(self, X):
        """Check whether the elements X are in the diadic partition."""
        normed_X = (X - self.__x_mins)/(self.__x_maxes - self.__x_mins)
        part_vals = np.floor(normed_X*(2**self.__depth)).astype(int)
        l_parts_nr = {tuple(p) for p, _ in self._l_parts}
        return np.array([tuple(p) in l_parts_nr for p in part_vals])

    def plot(self, region, *argv, **kwargs):
        raise Exception("TODO: Not yet implemented.")

    def cut_region(self, region):
        raise Exception("TODO: Not yet implemented.")

    def plot_diadic_partition(self, *argv, **kwargs):
        """Plots a dyadic partition, whithout using the region strategy."""
        parts = self._l_parts
        patch_col = list()
        for p in parts:
            disc_coords = p[0] # Discreete coordinates
            pos = [((disc_coords[i])/2**self.__depth)*(
                self.__x_maxes[i] - self.__x_mins[i]) + self.__x_mins[i]
                   for i in range(0, 2)]
            width, height = [(self.__x_maxes[i]-self.__x_mins[i])
                             /2**self.__depth
                             for i in range(0, 2)]
            patch_col.append(plt.Rectangle(pos, width, height))
        pc = PatchCollection(patch_col, *argv, **kwargs)
        plt.gca().add_collection(pc)

    # --------------- Private methods ---------------

    def __size_rectangle(self, i, dim):
        return ((self.__x_maxes[dim]-self.__x_mins[dim])
                *(i + 1)/2**self.__depth
                + self.__x_mins[dim])

    def __get_uniform_diadic_partition(self, X, y):
        """
            Splits the data into uniform diadic partitions.

            :param X:      Position of each point.
            :param y:      Binary class of the point, in {0,1}.

            :return:       A list of tuples (part_indices, (alpha, beta)).
        """

        self.__x_maxes = X.max(axis=0) + self.__eps
        self.__x_mins = X.min(axis=0) - self.__eps

        def get_diadic_split(cur_ys, cur_x_cs, cur_i, rem_dims, cur_dim):
            """
                Returns the partition info for the first and lasts component.

                Is based on a call to rec_diadic_split which wil call
                get_diadic_split again.
                :param cur_i:       Indice of the diadic split.
                :param rem_dims:    # of remaining dimensions.
                :param cur_dim:     Current dimension #.
            """
            if rem_dims == 1:
                n_pos_part = np.sum(cur_ys)
                n_part = len(cur_ys)
                partitions = [([cur_i], ((n_part-n_pos_part), n_pos_part))]
            else:
                # Call to split the remaining coordinates.
                supart_list = rec_diadic_split(
                    np.array(cur_x_cs), np.array(cur_ys), cur_dim+1)
                partitions = [([cur_i] + ind_list, n_points)
                              for ind_list, n_points in supart_list]
            cur_x_cs = list()
            cur_ys = list()
            return partitions

        def rec_diadic_split(X, y_in, cur_dim):
            """
                Recursive function that forms the partitions.

                Returns the indices for the partitions over the first
                dimension of X, i.e. X[:,0] and call itself recursively
                to split the remaining dimensions X[:,1] in each
                partition of X[:,1].
            """
            # We sort the elements according to the first dimension.
            ind_sort = np.argsort(X[:, 0])
            sorted_X = X[ind_sort, :]
            arr_y = y_in[ind_sort]
            # We split the X dimensions that are partitioned now and others.
            arr_x = sorted_X[:, 0]
            arr_x_c = sorted_X[:, 1:]

            partitions = list()
            cur_i = 0
            cur_x_cs = list()
            cur_ys = list()
            for x, y, x_c in zip(arr_x, arr_y, arr_x_c):
                # If the element x is not in the current partition:
                if x >= self.__size_rectangle(cur_i, cur_dim):
                    # If the partition is not empty, we save it:
                    if cur_ys:
                        partitions += get_diadic_split(cur_ys, cur_x_cs,
                                                       cur_i, X.shape[1],
                                                       cur_dim)
                        cur_x_cs = list()
                        cur_ys = list()
                    # If several rectangles do not contain anything:
                    while x >= self.__size_rectangle(cur_i, cur_dim):
                        cur_i += 1
                cur_x_cs.append(x_c)
                cur_ys.append(y)
            if cur_ys:
                partitions += get_diadic_split(cur_ys, cur_x_cs,
                                               cur_i, X.shape[1], cur_dim)
            return partitions
        return rec_diadic_split(X, y, 0)
