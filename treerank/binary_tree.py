"""
    Simple binary tree that splits the space.
"""
import numpy as np

class BinaryTree:
    """Implements a very simple binary tree."""
    def __init__(self, parent=None):
        # All attributes are used directly in the treerank implementations.
        self.left = None
        self.right = None
        self.data = dict()
        self.splitter = None
        self.weight = 1
        self.parent = parent

    def is_terminal_node(self):
        """Check whether the node has no children."""
        return (not self.right.data or self.right.weight == 0) and (
            not self.left.data or self.left.weight == 0)

    def print(self):
        """Prints the binary tree, RN is right node, LN left node."""
        def rec(prefix, tree):
            if tree.data:
                print(prefix + str(tree.data))
            if tree.left is not None:
                rec(prefix + "# LN #", tree.left)
            else:
                print(prefix + "# LN empty #")
            if tree.right is not None:
                rec(prefix + "# RN #", tree.right)
            else:
                print(prefix + "# RN empty #")
        print("Begin print tree:")
        rec("", self)

    def depth(self):
        """Computes the depth of the binary tree."""
        def rec(tree):
            if tree.right is None and tree.left is None:
                return 0
            if tree.right is None:
                return 1 + rec(tree.left)
            if tree.left is None:
                return 1 + rec(tree.right)
            return 1 + max(rec(tree.right), rec(tree.left))
        return rec(self)

    def n_leaves(self):
        """Computes the number of leaves of the binary tree."""
        def rec(tree):
            if not tree.data:
                return 1
            return rec(tree.right) + rec(tree.left)
        return rec(self)

    def delta_auc(self):
        """Computes the AUC delta for a node."""
        alpha_min, beta_min = self.data["alpha_min"], self.data["beta_min"]
        alpha_mid, beta_mid = self.data["alpha_mid"], self.data["beta_mid"]
        alpha_max, beta_max = self.data["alpha_max"], self.data["beta_max"]
        return ((alpha_max-alpha_min)*(beta_mid-beta_min)
                -(beta_max-beta_min)*(alpha_mid-alpha_min))

    def plot_2d(self, region, changing_color=True):
        """
            Plots the splitting of a 2D space done by the binary Tree.

            The splitter in the tree is required to be a HorizontalSplitter.
        """
        red = np.array([1., 0., 0.])
        green = np.array([0., 1., 0.])
        # print(debug)
        def rec(tree, d, region, color=np.array([0., 0., 0.])):
            # print(d)
            if tree and tree.data:
                # Plot the rectangle and define the sub-rectangles of children:
                if changing_color:
                    tree.splitter.plot(region, color=color)
                else:
                    tree.splitter.plot(region, color="black")
                ls_region, rs_region = tree.splitter.cut_region(region)
                rec(tree.right, d+1, rs_region, color + red*(2**(-d-1)))
                rec(tree.left, d+1, ls_region, color+green*(2**(-d-1)))
        rec(self, 0, region)

    def plot_2d_leaves(self, region, alpha=0.5):
        """
            Plots the splitting of a 2D space done by the binary Tree.

            The splitter in the tree is required to be a HorizontalSplitter.
        """
        red = np.array([1., 0., 0.])
        green = np.array([0., 1., 0.])
        # print(debug)
        def rec(tree, d, region, color=np.array([0., 0., 0.])):
            # print(d)
            if tree and tree.data:
                # Plot the rectangle and define the sub-rectangles of children:
                ls_region, rs_region = tree.splitter.cut_region(region)
                rec(tree.right, d+1, rs_region, color + red*(2**(-d-1)))
                rec(tree.left, d+1, ls_region, color + green*(2**(-d-1)))
            else:
                region.plot_patch(color=color, alpha=alpha)
        rec(self, 0, region)
