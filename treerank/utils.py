"""
    Utilities.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
# import ipdb; ipdb.set_trace()
POINTWISE_FREQ = 1000

def ptwise_roc(alphas, betas, target_alphas=np.linspace(1/POINTWISE_FREQ,
                                                        1-1/POINTWISE_FREQ,
                                                        POINTWISE_FREQ-2)):
    """Returns the pointwise ROC value at alpha."""
    i = 0
    prec_alpha = 0
    prec_beta = 0
    n_tot = len(target_alphas)
    cur_target = target_alphas[i]
    res_beta = list()
    # print("NEW TARGET: ", cur_target)
    for alpha, beta in zip(alphas, betas):
        # print("alpha ", alpha, "beta ", beta)
        # print("prec_alpha ", prec_alpha, "prec_beta ", prec_beta)
        while alpha > cur_target and i < n_tot-1:
            slope = (beta-prec_beta)/(alpha-prec_alpha)
            res_beta.append(prec_beta + slope*(cur_target - prec_alpha))
            i += 1
            cur_target = target_alphas[i]
            # print("NEW TARGET: ", cur_target)
        prec_alpha = alpha
        prec_beta = beta
    assert i == (n_tot - 1)
    return res_beta

def roc_curve(s, y):
    """Computes the ROC curve, works with equalities in score values."""
    s = np.array(s)
    y = np.array(y)
    ind_sort = np.argsort(s)
    s_sorted = s[ind_sort]
    y_sorted = y[ind_sort]
    n_pos = np.sum(y)
    n_tot = y.shape[0]
    n_neg = n_tot - n_pos
    s_vals = np.unique(s_sorted)
    res_roc = [(1, 1)]
    cur_point = (1, 1)
    i_s = 0
    for cur_s in s_vals:
        cur_pos, cur_neg = 0, 0
        while i_s <= n_tot-1 and s_sorted[i_s] == cur_s:
            cur_pos += y_sorted[i_s]
            cur_neg += 1 - y_sorted[i_s]
            i_s += 1
        cur_point = (cur_point[0] - cur_neg/n_neg,
                     cur_point[1] - cur_pos/n_pos)
        res_roc.append(cur_point)
    alphas, betas = list(zip(*res_roc))
    return alphas[::-1], betas[::-1]

def det_curve(s, y):
    """Computes the DET curve."""
    alpha, beta = roc_curve(s, y)
    return [alpha, list(1-np.array(beta))]

def auc(s, y):
    """Computes the AUC for scores s, classes y."""
    ind_sort = np.argsort(s)[::-1] # Highest to lowest
    s_sorted = s[ind_sort]
    y_sorted = y[ind_sort]
    # print(s_sorted, y_sorted)

    hs_n_pos, hs_n_neg = 0, 0
    n_eq_pos, n_eq_neg = 0, 0
    former_s = s_sorted[0]
    cur_eq_pairs = 0
    cur_gt_pairs = 0
    cur_lt_pairs = 0
    # print("unique values in s_sorted", np.unique(s_sorted))
    for cur_s, cur_y in zip(s_sorted, y_sorted):
        # print("cur_s ", cur_s, " cur_y ", cur_y, " former_s ", former_s)
        if cur_s != former_s:
            # print(("cur_s {} / cur_eq_pairs {} "
            #        + "/ cur_gt_pairs {} / cur_lt_pairs {}").format(
            #            cur_s, cur_eq_pairs, cur_gt_pairs, cur_lt_pairs))
            # print("n_eq_pos {} / n_eq_neg {}".format(n_eq_pos, n_eq_neg))
            former_s = cur_s
            cur_eq_pairs += n_eq_pos*n_eq_neg
            cur_gt_pairs += n_eq_neg*hs_n_pos
            cur_lt_pairs += n_eq_pos*hs_n_neg
            hs_n_pos += n_eq_pos
            hs_n_neg += n_eq_neg
            n_eq_pos, n_eq_neg = 0, 0
        if cur_y == 1:
            n_eq_pos += 1
        else:
            n_eq_neg += 1
    cur_eq_pairs += n_eq_pos*n_eq_neg
    cur_gt_pairs += n_eq_neg*hs_n_pos
    cur_lt_pairs += n_eq_pos*hs_n_neg
    n_tot_pairs = cur_eq_pairs + cur_gt_pairs + cur_lt_pairs
    return (cur_eq_pairs/2 + cur_gt_pairs)/n_tot_pairs

def plot_and_save(outname, fun, figsize=(4, 4)):
    """Returns a function that saves the plot generated by fun."""
    def plot_and_save_calls(*args, **kwargs):
        plt.figure(figsize=figsize)
        fun(*args, **kwargs)
        plt.legend()
        plt.grid()
        plt.savefig(outname, format="pdf")
    return plot_and_save_calls

def add_data_plot(X, y, set_lim=False, tree=None, alpha=0.5, turn_it=False):
    """Plots the data, as well as a SplitSpaceBinaryTree if it is provided."""
    #pylint: disable-msg=too-many-arguments
    if turn_it:
        X_tr = np.zeros(X.shape)
        X_tr[:, 0] = (X[:, 0] - X[:, 1])/np.sqrt(2)
        X_tr[:, 1] = (X[:, 0] + X[:, 1])/np.sqrt(2)
        X = X_tr
    plt.scatter(X[y == 0, 0], X[y == 0, 1],
                label="neg", color="red", alpha=alpha)
    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                label="pos", color="green", alpha=alpha)
    xlim, ylim = plt.xlim(), plt.ylim()
    if tree is not None:
        tree.plot_2d(xlim, ylim)
    if set_lim:
        plt.xlim(xlim)
        plt.ylim(ylim)

def plot_01_square(fun, gran=100):
    """Plots a function values on the [0, 1]x[0, 1] square."""
    x = np.linspace(0, 1, gran+1).reshape((gran+1, 1))
    X = np.hstack([x]*(gran+1))
    all_points = np.array([X.transpose().ravel(), X.ravel()]).transpose()
    z = fun(all_points)
    Z = z.reshape(gran+1, gran+1)
    plt.imshow(Z, origin="lower", cmap='gray', extent=[0, 1, 0, 1])
    # , interpolation='bilinear')
    plt.colorbar()

def get_random_pairs(X, y, n_pairs):
    """Returns n_pairs random pairs from X,y."""
    # TODO: Check whether I need to try and rebalance the classes.
    n = y.shape[0]
    X_pairs = list()
    z_pairs = list()
    for _ in range(n_pairs):
        inds = np.random.randint(0, n, 2)
        X_pairs.append(np.hstack([X[inds[0]], X[inds[1]]]))
        z_pairs.append(int(y[inds[0]] == y[inds[1]]))
    X_pairs, z_pairs = np.array(X_pairs), np.array(z_pairs)
    n_pos = (z_pairs == 1).sum()
    n_neg = z_pairs.shape[0] - n_pos
    logging.debug("For %s pairs I have %s negatives and %s positives",
                  n_pairs, n_neg, n_pos)
    return X_pairs, z_pairs


# --------------- Data generation ---------------

# 1-dimensional
DEFAULT_A = 0.80

def mu_1(X, a=DEFAULT_A):
    """Distribution of the simrank paper."""
    mu = np.zeros(X.shape)
    filt = X <= 0.5
    mu[filt] = 1 - np.power(1 - 2*X[filt], (1-a)/a)
    mu[~filt] = 1 + np.power(2 - 2*X[~filt], (1-a)/a)
    return mu

def simrank_data_dist(n=100, a=DEFAULT_A):
    """Data from the simrank paper."""
    X = np.random.uniform(0, 1, (n, 1))
    y = np.random.binomial(1, 0.5*mu_1(X, a=a).ravel())
    return X, y

def quadratic_data_dist(n=100):
    """Data from the simrank paper."""
    X = np.random.uniform(0, 1, (n, 1))
    y = np.random.binomial(1, 4*np.power(X-0.5, 2).ravel())
    return X, y

CHECKERBOARD_SIZE = 2
CHECKERBOARD_PREC = 0.90

def checkerboard_data_dist(n=100):
    """Data from the simrank paper."""
    X = np.random.uniform(0, 1, (n, 1))
    inter = (np.floor(X*CHECKERBOARD_SIZE) % 2)
    eta = (2*CHECKERBOARD_PREC-1)*inter + (1-CHECKERBOARD_PREC)
    y = np.random.binomial(1, eta.ravel())
    return X, y

# 2-dimensional
W_VAL = np.array([[4/7], [2/7]])

def oblique_data_distr(n=800):
    """Recreates data from Fig. 10.a) in [1]."""
    X = np.random.uniform(0, 1, (n, 2))
    eta = X.dot(W_VAL)
    y = np.random.binomial(1, eta.ravel())
    return X, y

def eta_turning_circle(X):
    """Eta=P(Y=+1|X=x) for the turning data example."""
    return 1 - 2*(np.arctan(X[:, 1] / (X[:, 0]))/np.pi)

def gen_data_turning_circle(n):
    """Generates data in r<=1 such that eta depends of the angle."""
    X = list()
    for _ in range(0, n):
        x_prop = np.array([1, 1])
        while np.linalg.norm(x_prop) > 1:
            x_prop = np.random.uniform(0, 1, 2)
        X.append(x_prop)
    X = np.array(X)
    Y = np.random.binomial(n=1, p=eta_turning_circle(X))
    return X, Y

# -------- Transformations for the symmetric region -------

def transform_sym(X):
    """Transforms coordinates from symmetric basis to canonical basis."""
    d = X.shape[1]//2
    # print(X.shape)
    X0 = X[:, :d]
    X1 = X[:, d:]
    r = np.random.binomial(1, 0.5, (X.shape[0], 1))
    coord1 = r*((X0 + X1)/np.sqrt(2)) + (1-r)*((-X0+X1)/np.sqrt(2))
    coord2 = (1-r)*((X0 + X1)/np.sqrt(2)) + r*((-X0+X1)/np.sqrt(2))
    return np.hstack([coord1, coord2])

def dir_transform_sym(X):
    """Transforms coordinates from canonical basis to symmetric basis."""
    d = X.shape[1]//2
    X0 = X[:, :d]
    X1 = X[:, d:]
    return np.hstack([np.abs((X0 - X1))/np.sqrt(2), (X0 + X1)/np.sqrt(2)])

# --------------- Notes ---------------

# # To get a figure without whitespace around:
# from matplotlib.ticker import NullLocator
# plt.axis('off')
# plt.gca().xaxis.set_major_locator(NullLocator())
# plt.gca().yaxis.set_major_locator(NullLocator())
# plt.savefig("plot.png", format="png",
#             pad_inches=0, bbox_inches='tight')