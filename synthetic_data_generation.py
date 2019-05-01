"""Similarity tree generating function."""
import numpy as np
from treerank.utils import transform_sym, dir_transform_sym
from treerank.binary_tree import BinaryTree
from treerank.split_space.diag_sym_splitter import DiagSymSplitter

def generate_tree(dim=2, depth=6, gamma=0.8, p=0.5):
    """
        Generates a random fully grown tree.

        dim is the dim of the pair.
    """
    leaves = list()
    k_leaf = [0]
    def attach_random_nodes(d, lims):
        tree = BinaryTree()
        if d == 0:
            tree.lims = lims
            leaves.append(tree)
            tree.k_leaf = k_leaf[0]
            k_leaf[0] = k_leaf[0] + 1
            # print("limits",
            #       [["{:.2f}".format(x), "{:.2f}".format(y)]
            #        for x, y in tree.lims])
            return tree
        tree.split_dim = np.random.randint(dim)
        # Possible to split while favoring the first dimensions
        tree.split_val = np.random.uniform(*lims[tree.split_dim])
        # print(tree.split_dim, tree.split_val)
        tree.split_orient = np.random.binomial(1, 0.5)

        upper_corner = [lims[i] if i != tree.split_dim
                        else [tree.split_val, lims[i][1]]
                        for i in range(dim)]
        lower_corner = [lims[i] if i != tree.split_dim
                        else [lims[i][0], tree.split_val]
                        for i in range(dim)]

        if tree.split_orient:
            left_lims = upper_corner
            right_lims = lower_corner
        else:
            left_lims = lower_corner
            right_lims = upper_corner

        tree.left = attach_random_nodes(d-1, left_lims)
        tree.right = attach_random_nodes(d-1, right_lims)
        return tree

    lims = [[0, 1]]*(dim//2) + [[-1, 1]]*(dim//2)
    tree = attach_random_nodes(depth, lims)

    tree.gamma = gamma
    tree.leaves = leaves
    tree.n_leaves = len(leaves) # L
    l_tot = tree.n_leaves

    tree.delta_l_pos = np.array([gamma**(-l/l_tot)
                                 for l in range(0, tree.n_leaves)])
    tree.delta_l_pos = tree.delta_l_pos/np.sum(tree.delta_l_pos)
    tree.delta_l_neg = np.array([gamma**(l/l_tot)
                                 for l in range(0, tree.n_leaves)])
    tree.delta_l_neg = tree.delta_l_neg/np.sum(tree.delta_l_neg)

    arg_sort = np.argsort(tree.delta_l_pos/tree.delta_l_neg)[::-1]
    sorted_pos = tree.delta_l_pos[arg_sort]
    sorted_neg = tree.delta_l_neg[arg_sort]

    tree.auc = (np.sum(sorted_pos*sorted_neg)/2
                + np.triu(sorted_pos.reshape((-1, 1)).dot(
                    sorted_neg.reshape((1, -1))), k=1).sum())

    resort = np.array(range(tree.n_leaves))
    np.random.shuffle(resort)
    tree.delta_l_pos = tree.delta_l_pos[resort]
    tree.delta_l_neg = tree.delta_l_neg[resort]

    # print("AUC = {:.2f}".format(tree.auc))

    tree.p = p
    tree.dim = dim
    tree.depth = dim
    return tree

def generate_sample_from_tree(n_elems, tree):
    """Generates a sample from a tree."""
    z = list()
    X = list()
    # print("neg probs:", tree.delta_l_neg.cumsum())
    # print("pos probs:", tree.delta_l_pos.cumsum())
    for _ in range(n_elems):
        cur_z = np.random.binomial(1, tree.p)
        z.append(cur_z)
        if cur_z == 0:
            proba_leaf = tree.delta_l_neg
        else:
            proba_leaf = tree.delta_l_pos
        leaf = np.argmax(np.random.uniform() < proba_leaf.cumsum())
        x = np.array([np.random.uniform(x_min, x_max)
                      for x_min, x_max in tree.leaves[leaf].lims])
        X.append(x)
    X = np.array(X)
    X = transform_sym(X)
    z = np.array(z)
    return X, z

def simtree_eta(tree, X):
    """Returns the probability distribution on the data X."""
    stacked_X = dir_transform_sym(X)

    def compute_score(cur_X, cur_tree):
        if hasattr(cur_tree, 'split_dim'):
            s = np.zeros(cur_X.shape[0])
            filt = ((cur_X[:, cur_tree.split_dim] > cur_tree.split_val)
                    == cur_tree.split_orient)
            s[filt] = compute_score(cur_X[filt], cur_tree.left)
            s[~filt] = compute_score(cur_X[~filt], cur_tree.right)
            return s
        delta_pos = tree.delta_l_pos[cur_tree.k_leaf]
        delta_neg = tree.delta_l_neg[cur_tree.k_leaf]
        return (tree.p*delta_pos)/(tree.p*delta_pos + (1-tree.p)*delta_neg)
    return compute_score(stacked_X, tree)

def simtree_optimal_roc(tree):
    """Returns the optimal ROC for a tree."""
    slopes = tree.delta_l_pos/tree.delta_l_neg
    ind_sorted_slopes = np.argsort(slopes)[::-1]
    alphas, betas = [0], [0]
    cur_alpha, cur_beta = 0, 0
    for k in ind_sorted_slopes:
        cur_alpha += tree.delta_l_neg[k]
        cur_beta += tree.delta_l_pos[k]
        alphas.append(cur_alpha)
        betas.append(cur_beta)
    cur_alpha, cur_beta = 1, 1
    alphas.append(cur_alpha)
    betas.append(cur_beta)
    return alphas, betas
