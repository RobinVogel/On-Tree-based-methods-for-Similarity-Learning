"""
    Creates a RectRegion class that inherits from Region (see region.py)
    that represent a rectangular region in the symmetric basis.
"""
import numpy as np
import matplotlib.pyplot as plt

from .rect_region import RectRegion
from .utils import transform_sym

class SymRegion(RectRegion):
    """
        Rectangular region.

        Defined by x_lim and y_lim, coordinates in the symmetric basis.
        Hence, it uses all the same splitting functions as the RectRegion,
        but different plotting rules.
    """
    def __init__(self, x_lim, y_lim, canonical_plot=True):
        super(SymRegion, self).__init__(x_lim, y_lim)
        self.canonical_plot = canonical_plot

    def plot_bounds(self, *argv, **kwargs):
        """Plots the bounds of the region."""
        polygon = np.array([[max(0, self.x_lim[0]), self.y_lim[0]],
                            [self.x_lim[1], self.y_lim[0]],
                            [self.x_lim[1], self.y_lim[1]],
                            [max(0, self.x_lim[0]), self.y_lim[1]]])
        last_point = polygon[-1]
        for x, y in polygon:
            self.__plot_sym_line([last_point[0], x],
                                 [last_point[1], y], *argv, **kwargs)
            last_point = [x, y]

    def plot_patch(self, *argv, **kwargs):
        """Plots the patch defined by the region."""
        polygon = np.array([[max(0, self.x_lim[0]), self.y_lim[0]],
                            [self.x_lim[1], self.y_lim[0]],
                            [self.x_lim[1], self.y_lim[1]],
                            [max(0, self.x_lim[0]), self.y_lim[1]]])
        self.__plot_sym_polygon(polygon, *argv, **kwargs)

    def plot_coord_split(self, cut_val, cut_dim, *argv, **kwargs):
        """Plots the line defined by cut_val and cut_dim."""
        if cut_dim == 0:
            # Plot on x-y/2
            if self.x_lim[0] <= cut_val <= self.x_lim[1]:
                pts = np.array([[cut_val]*2, self.y_lim])
            else:
                return None
        else:
            if self.y_lim[0] <= cut_val <= self.y_lim[1]:
                if self.x_lim[0] <= 0 <= self.x_lim[1]:
                    x_lim_l1 = [0, np.max(self.x_lim)]
                    pts = np.array([x_lim_l1, [cut_val]*2])
                else:
                    x_lim_min = np.min(np.abs(self.x_lim))
                    x_lim_max = np.max(np.abs(self.x_lim))
                    pts = np.array([[x_lim_min, x_lim_max], [cut_val]*2])
            else:
                return None

        self.__plot_sym_line(pts[0], pts[1], *argv, **kwargs)
        return None

    # --------------- Protected interface ---------------

    def _gen_sub(self, x_lim, y_lim):
        """Generates a sub region, parameterized in the symmetric basis."""
        return SymRegion(x_lim, y_lim, self.canonical_plot)

    # --------------- Private interface ---------------

    def __plot_sym_line(self, xs, ys, *argv, **kwargs):
        if self.canonical_plot:
            x1, y1 = transform_sym(xs[0], ys[0])
            x2, y2 = transform_sym(xs[1], ys[1])
        else:
            x1, y1 = xs[0], ys[0]
            x2, y2 = xs[1], ys[1]
        plt.plot([x1, x2], [y1, y2], *argv, **kwargs)
        if self.canonical_plot:
            x1, y1 = transform_sym(-xs[0], ys[0])
            x2, y2 = transform_sym(-xs[1], ys[1])
        else:
            x1, y1 = -xs[0], ys[0]
            x2, y2 = -xs[1], ys[1]
        plt.plot([x1, x2], [y1, y2], *argv, **kwargs)

    def __plot_sym_polygon(self, polygon, *argv, **kwargs):
        if self.canonical_plot:
            tr_polygon = np.array([transform_sym(x, xp) for x, xp in polygon])
        else:
            tr_polygon = polygon
        plt.gca().fill(tr_polygon[:, 0], tr_polygon[:, 1], *argv, **kwargs)
        polygon[:, 0] = - polygon[:, 0]
        if self.canonical_plot:
            polygon = np.array([transform_sym(x, xp) for x, xp in polygon])
        plt.gca().fill(polygon[:, 0], polygon[:, 1], *argv, **kwargs)
