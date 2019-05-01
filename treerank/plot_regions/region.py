"""
    Creates a Region class whose aim is to represent the acceptance
    regions of the LeafRank algorithm, when they belong to a 2D space.
"""
from abc import ABC, abstractmethod

class Region(ABC):
    """Regions for plotting the space splitters."""

    # --------------- Init and copy methods ---------------

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.is_union_region = False
        # True if we represent the splitting regions in the canonical basis.
        self.canonical_plot = True

    # --------------- Region plotting methods ---------------

    @abstractmethod
    def plot_patch(self, *argv, **kwargs):
        """Plots a patch corresponding to the region."""

    @abstractmethod
    def plot_bounds(self, *argv, **kwargs):
        """Plots the bounds of the region."""

    @abstractmethod
    def plot_coord_split(self, cut_val, cut_dim, *argv, **kwargs):
        """Plots the separating line defined by cut_val, cut_dim in region."""

    @abstractmethod
    def plot_cross_split(self, cut_val, *argv, **kwargs):
        """Plots the separating cross defined by cut_val, cut_dim in region."""

    # --------------- Region splitting methods ---------------

    @abstractmethod
    def split_coord(self, cut_dim, cut_val):
        """Returns the regions from splitting the region with a line."""

    @abstractmethod
    def split_cross(self, cut_val):
        """Returns the regions from splitting the region with a cross."""
