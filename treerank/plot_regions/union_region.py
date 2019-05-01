"""
    Defines a UnionRegion defined as the union of other regions.
    See region.py for the base class region.
"""
from .region import Region

class UnionRegion(Region):
    """Region defined as the union of other regions."""

    # --------------- Init and copy methods ---------------

    def __init__(self):
        self.regions = list()
        super(UnionRegion).__init__()
        self.is_union_region = True

    @classmethod
    def from_regions(cls, region):
        """Defines an UnionRegion from a region or list of regions."""
        res = cls()
        if isinstance(region, (list,)):
            res.regions = region
        else:
            res.regions = [region]
        return res

    # --------------- Region plotting methods ---------------

    def plot_patch(self, *argv, **kwargs):
        """Plots the patch defined by the region."""
        for region in self.regions:
            region.plot_patch(*argv, **kwargs)

    def plot_bounds(self, *argv, **kwargs):
        """Plots the bounds of the region."""
        for region in self.regions:
            region.plot_bounds(*argv, **kwargs)

    def plot_coord_split(self, cut_val, cut_dim, *argv, **kwargs):
        """Plots the line defined by cut_val and cut_dim."""
        for region in self.regions:
            region.horiz_plot(cut_val, cut_dim, *argv, **kwargs)

    def plot_cross_split(self, cut_val, *argv, **kwargs):
        """Plots the separating cross defined by cut_val, cut_dim in region."""
        for region in self.regions:
            region.cross_plot(cut_val, *argv, **kwargs)

    # --------------- Region splitting methods ---------------

    def split_coord(self, cut_dim, cut_val):
        """Returns the regions from splitting the region with a line."""
        l_res_reg = UnionRegion()
        r_res_reg = UnionRegion()
        for region in self.regions:
            l_reg, r_reg = region.split_coord(cut_dim, cut_val)
            if l_reg.is_union_region:
                l_res_reg.regions += l_reg.regions
            else:
                l_res_reg.regions.append(l_reg)
            if r_reg.is_union_region:
                r_res_reg.regions += r_reg.regions
            else:
                r_res_reg.regions.append(r_reg)
        return l_res_reg, r_res_reg

    def split_cross(self, cut_val):
        """Returns the regions from splitting the region with a cross."""
        l_res_reg = UnionRegion()
        r_res_reg = UnionRegion()
        for region in self.regions:
            l_reg, r_reg = region.split_cross(cut_val)
            if l_reg.is_union_region:
                l_res_reg.regions += l_reg.regions
            else:
                l_res_reg.regions.append(l_reg)
            if r_reg.is_union_region:
                r_res_reg.regions += r_reg.regions
            else:
                r_res_reg.regions.append(r_reg)
        return l_res_reg, r_res_reg
