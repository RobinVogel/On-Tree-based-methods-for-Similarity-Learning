"""
    Creates a RectRegion class that inherits from Region (see region.py)
    that represent a rectangular region in the canonical basis.
"""
import numpy as np
import matplotlib.pyplot as plt

from .region import Region
from .union_region import UnionRegion

class RectRegion(Region):
    """
        Rectangular region.

        Defined by x_lim and y_lim, coordinates in the canonical basis.
    """

    # --------------- Init and copy methods ---------------

    def __init__(self, x_lim, y_lim):
        self.x_lim = x_lim
        self.y_lim = y_lim
        super(RectRegion).__init__()

    @classmethod
    def from_region(cls, rect_region):
        """Initiates the RectRegion from another RectRegion."""
        return cls(rect_region.x_lim, rect_region.y_lim)

    # --------------- Region plotting methods ---------------

    def plot_patch(self, *argv, **kwargs):
        """Plots the patch defined by the region."""
        points_polygon = [[self.x_lim[0], self.y_lim[0]],
                          [self.x_lim[1], self.y_lim[0]],
                          [self.x_lim[1], self.y_lim[1]],
                          [self.x_lim[0], self.y_lim[1]]]

        points_polygon = np.array(points_polygon)

        plt.gca().fill(points_polygon[:, 0], points_polygon[:, 1],
                       *argv, **kwargs)

    def plot_bounds(self, *argv, **kwargs):
        """Plots the bounds of the region."""
        plt.plot(self.x_lim, [self.y_lim[0]]*2, *argv, **kwargs)
        plt.plot(self.x_lim, [self.y_lim[1]]*2, *argv, **kwargs)
        plt.plot([self.x_lim[0]]*2, self.y_lim, *argv, **kwargs)
        plt.plot([self.x_lim[1]]*2, self.y_lim, *argv, **kwargs)

    def plot_coord_split(self, cut_val, cut_dim, *argv, **kwargs):
        """Plots the line defined by cut_val and cut_dim."""
        if cut_dim == 0:
            if self.x_lim[0] <= cut_val <= self.x_lim[1]:
                plt.plot([cut_val]*2, self.y_lim, *argv, **kwargs)
        else:
            if self.y_lim[0] <= cut_val <= self.y_lim[1]:
                plt.plot(self.x_lim, [cut_val]*2, *argv, **kwargs)

    def plot_cross_split(self, cut_val, *argv, **kwargs):
        """Plots the separating cross defined by cut_val, cut_dim in region."""
        if self.x_lim[0] <= cut_val <= self.x_lim[1]:
            plt.plot([cut_val]*2, self.y_lim, *argv, **kwargs)
        if self.y_lim[0] <= cut_val <= self.y_lim[1]:
            plt.plot(self.x_lim, [cut_val]*2, *argv, **kwargs)

    # --------------- Region splitting methods ---------------

    def split_coord(self, cut_dim, cut_val):
        """Returns the regions from splitting the region with a line."""
        if cut_dim == 0:
            if self.x_lim[0] <= cut_val <= self.x_lim[1]:
                x_lim_u, y_lim_u = [cut_val, self.x_lim[1]], self.y_lim
                x_lim_l, y_lim_l = [self.x_lim[0], cut_val], self.y_lim
            elif self.x_lim[0] > cut_val:
                x_lim_u, y_lim_u = [self.x_lim[0], self.x_lim[1]], self.y_lim
                x_lim_l, y_lim_l = [self.x_lim[0], self.x_lim[0]], self.y_lim
            else:
                x_lim_u, y_lim_u = [self.x_lim[1], self.x_lim[1]], self.y_lim
                x_lim_l, y_lim_l = [self.x_lim[0], self.x_lim[1]], self.y_lim
        else:
            if self.y_lim[0] <= cut_val <= self.y_lim[1]:
                x_lim_u, y_lim_u = self.x_lim, [cut_val, self.y_lim[1]]
                x_lim_l, y_lim_l = self.x_lim, [self.y_lim[0], cut_val]
            elif self.y_lim[0] > cut_val:
                x_lim_u, y_lim_u = self.x_lim, [self.y_lim[0], self.y_lim[1]]
                x_lim_l, y_lim_l = self.x_lim, [self.y_lim[0], self.y_lim[0]]
            else:
                x_lim_u, y_lim_u = self.x_lim, [self.y_lim[1], self.y_lim[1]]
                x_lim_l, y_lim_l = self.x_lim, [self.y_lim[0], self.y_lim[1]]
        # Give the sub-rectangles to the children.
        return self._gen_sub(x_lim_u, y_lim_u), self._gen_sub(x_lim_l, y_lim_l)

    def split_cross(self, cut_val):
        """Returns the regions from splitting the region with a cross."""
        if (self.x_lim[0] <= cut_val <= self.x_lim[1]) and (
                self.y_lim[0] <= cut_val <= self.y_lim[1]):
            # Create two zones and return them
            upper_x, lower_x = self.split_coord(0, cut_val)
            upper_x_y, up_x_low_y = upper_x.split_coord(1, cut_val)
            low_x_up_y, lower_x_y = lower_x.split_coord(1, cut_val)
            part_diag = UnionRegion.from_regions([upper_x_y, lower_x_y])
            other_part = UnionRegion.from_regions([low_x_up_y, up_x_low_y])
        elif ((self.x_lim[0] >= cut_val) or (cut_val >= self.x_lim[1])) and (
                self.y_lim[0] <= cut_val <= self.y_lim[1]):
            # Split on the y axis only
            if self.x_lim[0] >= cut_val:
                part_diag, other_part = self.split_coord(1, cut_val)
            elif cut_val >= self.x_lim[1]:
                other_part, part_diag = self.split_coord(1, cut_val)
        elif (self.x_lim[0] <= cut_val <= self.x_lim[1]) and (
                (self.y_lim[0] >= cut_val) or (cut_val >= self.y_lim[1])):
            # Split on the x axis only
            if self.y_lim[0] >= cut_val:
                part_diag, other_part = self.split_coord(0, cut_val)
            elif cut_val >= self.y_lim[1]:
                other_part, part_diag = self.split_coord(0, cut_val)
        else:
            # Split on no axis
            if ((cut_val >= self.x_lim[1]) and (cut_val >= self.y_lim[1])) or (
                    (cut_val <= self.x_lim[0]) and (cut_val <= self.y_lim[0])):
                part_diag = self
                other_part = self._gen_sub([self.x_lim[0], self.x_lim[0]],
                                           [self.y_lim[0], self.y_lim[0]])
            else:
                part_diag = self._gen_sub([self.x_lim[0], self.x_lim[0]],
                                          [self.y_lim[0], self.y_lim[0]])
                other_part = self
        return part_diag, other_part

    # --------------- Protected interface ---------------

    @classmethod
    def _gen_sub(cls, x_lim, y_lim):
        """Generates a sub region, parameterized in the canonical basis."""
        return cls(x_lim, y_lim)
