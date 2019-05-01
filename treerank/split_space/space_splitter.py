"""
    Abstract class for a splitter of a space.
"""
from abc import ABC, abstractmethod
import numpy as np

class SpaceSplitter(ABC):
    r"""
        Abstract class that implements splits of the space, defined by a set.
    """
    # pylint: disable=too-many-instance-attributes

    @abstractmethod
    def __init__(self, fact_neg, fact_pos):
        self.fact_neg, self.fact_pos = fact_neg, fact_pos
        self.n_neg, self.n_pos = [0]*2
        self.best_value = None
        self.crit = None
        # Protected
        # Some splitters define their acceptance zone as an union of parts.
        self._l_parts = None # Left-side of tree parts
        self._r_parts = None # Right-side of tree parts

    @abstractmethod
    def fit(self, X, y):
        """Fits the space splitter."""

    @abstractmethod
    def check(self, X):
        """Gives a boolean that checks which elements of X are in the set."""

    @abstractmethod
    def plot(self, region, *argv, **kwargs):
        """Plots the region delimited by the splitter."""

    @abstractmethod
    def cut_region(self, region):
        """Return the regions in and out when the splits cuts the region."""

    #  --------------- Protected methods ---------------

    def _maximize_crit_from_parts(self, partitions):
        """
            Maximizes the empirical criterion described in [1].

            :param partitions:  List of (partitions, (n_neg, n_pos)).
            :param fact_neg:    Weight on negative local density.
            :param fact_pos:    Weight on positive local density.

            :return:            A triplet list giving the Lk's in L,
                                the Lk's in R and the criterion value.
        """
        emp_measure = [self.__emp_meas(v) for _, v in partitions]
        kept_parts = np.array(emp_measure) >= 0
        self._l_parts, self._r_parts = list(), list()
        for p, k in zip(partitions, kept_parts):
            if k:
                self._l_parts.append(p)
            else:
                self._r_parts.append(p)
        self.crit = np.sum([self.__emp_meas(v) for _, v in self._l_parts])

    #  --------------- Private methods ---------------

    def __emp_meas(self, vals):
        return self.fact_neg*vals[0] + self.fact_pos*vals[1]
