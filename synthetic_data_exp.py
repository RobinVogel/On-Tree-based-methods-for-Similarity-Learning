"""
    Simulated data experiments on trees.
"""
from datetime import datetime
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from synthetic_data_generation import (
    generate_tree,
    generate_sample_from_tree,
    simtree_optimal_roc,
    simtree_eta
)
from treerank.treerank_similarity import SimTreeRank
from treerank.split_space.diag_sym_splitter import DiagSymSplitter
from treerank.utils import (
    roc_curve,
    ptwise_roc,
    auc
)

N_TRAIN_FACTOR = 150
EXP_FACTOR = 1.25

# --------------- Experiments ---------------

def main_experiments():
    """ Experiments that form the synthetic data experiments of the paper ."""
    # Meta parameters
    n_exp = 400
    batch_exp_size = 10
    mod_print = batch_exp_size

    # run_exp("model complexity", model_complexity_exp,
    #         n_exp=n_exp, mod_print=mod_print)
    # run_exp("class asymmetry", class_asymmetry_exp,
    #         n_exp=n_exp, mod_print=mod_print)
    # run_exp("model bias", model_bias_exp,
    #         n_exp=n_exp, mod_print=mod_print)

    run_parallel_exp("model complexity", model_complexity_exp,
                     n_exp=n_exp, mod_print=mod_print,
                     batch_exp_size=batch_exp_size)
    run_parallel_exp("model bias", model_bias_exp,
                     n_exp=n_exp, mod_print=mod_print,
                     batch_exp_size=batch_exp_size)
    run_parallel_exp("class asymmetry", class_asymmetry_exp,
                     n_exp=n_exp, mod_print=mod_print,
                     batch_exp_size=batch_exp_size)

    print("all done - " + datetime.now().ctime())

# --------------- Definition of the experiments ---------------

def model_complexity_exp(n_test=10000, dim=6, gamma=0.01, p=0.5,
                         gt_depth_list=(1, 2, 3, 4, 5)):
    """Model complexity experiment, see the paper."""
    l_opt_auc, l_auc_diff, l_sup_diff = list(), list(), list()
    np.random.seed()
    for gt_depth in gt_depth_list:
        n_train = int(N_TRAIN_FACTOR*(EXP_FACTOR**(gt_depth**2)))
        opt_auc, auc_diff, sup_diff = experiment(
            gamma=gamma, dim=dim, p=p, gt_depth=gt_depth,
            n_train=n_train, n_test=n_test, depth=gt_depth)
        l_opt_auc.append(opt_auc)
        l_auc_diff.append(auc_diff)
        l_sup_diff.append(sup_diff)
    return l_opt_auc, l_auc_diff, l_sup_diff, gt_depth_list

def class_asymmetry_exp(n_test=100000, dim=6, gamma=0.01, gt_depth=3,
                        pos_prop_list=(1/2, 1/10, 1/1000, 1/5000)):
    """Class asymmetry experiment, see the paper."""
    n_train = int(N_TRAIN_FACTOR*(EXP_FACTOR**(gt_depth**2)))
    l_opt_auc, l_auc_diff, l_sup_diff = list(), list(), list()
    for p in pos_prop_list:
        opt_auc, auc_diff, sup_diff = experiment(
            gamma=gamma, dim=dim, p=p, gt_depth=gt_depth,
            n_train=n_train, n_test=n_test, depth=gt_depth)
        l_opt_auc.append(opt_auc)
        l_auc_diff.append(auc_diff)
        l_sup_diff.append(sup_diff)
    return l_opt_auc, l_auc_diff, l_sup_diff, pos_prop_list

def model_bias_exp(n_test=10000, dim=6, gamma=0.01, gt_depth=3, p=0.5,
                   depth_list=(1, 2, 3, 8)):
    """Model bias experiment, see the paper."""
    n_train = int(N_TRAIN_FACTOR*(EXP_FACTOR**(gt_depth**2)))
    l_opt_auc, l_auc_diff, l_sup_diff = list(), list(), list()
    for depth in depth_list:
        opt_auc, auc_diff, sup_diff = experiment(
            gamma=gamma, dim=dim, p=p, gt_depth=gt_depth,
            n_train=n_train, n_test=n_test, depth=depth)
        l_opt_auc.append(opt_auc)
        l_auc_diff.append(auc_diff)
        l_sup_diff.append(sup_diff)
    return l_opt_auc, l_auc_diff, l_sup_diff, depth_list

# --------------- Utilities ---------------

def run_exp(name_exp, fun_exp, n_exp=200, mod_print=10):
    """Run n_exp times the fun_exp and stores the results in a json file."""
    print("doing {} experiments - {}".format(name_exp, datetime.now().ctime()))
    d_res = {"opt_auc": list(),
             "auc_diff": list(),
             "sup_diff": list(),
             "param": list()}
    for i_exp in range(n_exp):
        if i_exp % mod_print == 0:
            print("done {} out of {} - ".format(i_exp, n_exp)
                  + datetime.now().ctime())
        opt_auc, auc_diff, sup_diff, param = fun_exp()
        d_res["opt_auc"] += opt_auc
        d_res["auc_diff"] += auc_diff
        d_res["sup_diff"] += sup_diff
        d_res["param"] += param
    with open("{}.json".format(name_exp.replace(" ", "_")), "wt") as f:
        json.dump(d_res, f)

def run_parallel_exp(name_exp, fun_exp,
                     n_exp=200, mod_print=10, batch_exp_size=25):
    """Do the same as run_exp but in a distributed manner."""
    print("Doing {} experiments - {}".format(
        name_exp, datetime.now().ctime()))
    d_res = {"opt_auc": list(),
             "auc_diff": list(),
             "sup_diff": list(),
             "param": list()}
    i_exp = 0
    last_print = - mod_print - 1
    while i_exp < n_exp:
        if i_exp - last_print > mod_print:
            print("Done {} out of {} - ".format(i_exp, n_exp)
                  + datetime.now().ctime())
            last_print = i_exp - 1
        cur_n_exp = min(batch_exp_size, n_exp - i_exp)
        pool = Parallel(n_jobs=n_exp)
        # Have the number of CPU on the machine: multiprocessing.cpu_count()
        out_values = pool(delayed(fun_exp)() for _ in range(batch_exp_size))
        for opt_auc, auc_diff, sup_diff, param in out_values:
            d_res["opt_auc"] += opt_auc
            d_res["auc_diff"] += auc_diff
            d_res["sup_diff"] += sup_diff
            d_res["param"] += param
        i_exp += cur_n_exp
    with open("{}.json".format(name_exp.replace(" ", "_")), "wt") as f:
        json.dump(d_res, f)

def experiment(gamma=0.01, dim=2, p=0.5, gt_depth=2, n_train=10000,
               n_test=100000, depth=5, log=False):
    """Runs a TreeRank fitting experiment, specified by function arguments."""
    PREFIX = "### "

    # Generate data:
    if log:
        print(PREFIX + "Generating the sample... - " + datetime.now().ctime())
    tree = generate_tree(dim=dim, depth=gt_depth, gamma=gamma, p=p)
    X_train, z_train = generate_sample_from_tree(n_train, tree)
    X_test, z_test = generate_sample_from_tree(n_test, tree)
    n_train_pos = z_train.sum()
    n_train_neg = z_train.shape[0] - n_train_pos
    # n_test_pos = z_test.sum()
    # n_test_neg = z_test.shape[0] - n_test_pos
    if n_train_pos == 0 or n_train_neg == 0:
        roc_model = ptwise_roc([0, 1], [0, 1])
        roc_optimal = ptwise_roc(*simtree_optimal_roc(tree))
        sup_diff = np.max(np.abs(np.array(roc_optimal) - np.array(roc_model)))
        return tree.auc, tree.auc - 0.5, sup_diff
    if log:
        print(PREFIX + "Done generating the sample... - "
              + datetime.now().ctime())

    # Fit the model:
    if log:
        print(PREFIX + "Fitting model... - " + datetime.now().ctime())
    gen_split = DiagSymSplitter
    model = SimTreeRank(gen_split, depth)
    model.fit_pairs(X_train, z_train)
    if log:
        print(PREFIX + "Done fitting model... - " + datetime.now().ctime())

    # Score the pairs:
    if log:
        print(PREFIX + "Scoring the pairs... - " + datetime.now().ctime())
    s_test = model.score(X_test)
    if log:
        print(PREFIX + "Done scoring the pairs... - " + datetime.now().ctime())

    # Compute the results:
    if log:
        print(PREFIX + "Computing pointwise ROCs... - "
              + datetime.now().ctime())
    roc_optimal = ptwise_roc(*simtree_optimal_roc(tree))
    roc_model = ptwise_roc(*roc_curve(s_test, z_test))
    sup_diff = np.max(np.abs(np.array(roc_optimal) - np.array(roc_model)))

    if log:
        print(PREFIX + "Computing AUC... - " + datetime.now().ctime())
    auc_model = auc(s_test, z_test)

    auc_diff = tree.auc - auc_model
    if log:
        print(PREFIX + "Done exp... - " + datetime.now().ctime())
    return tree.auc, auc_diff, sup_diff


# --------------- Post processing ---------------

def data_analysis(json_name, variant="normal"): # or quantile
    """Performs data analysis and summarizes in a nice latex table."""
    df = pd.DataFrame(json.load(open(json_name, "rt")))
    alpha = 0.05
    uniq_param = df["param"].unique()
    d_res = {"param": list(), "auc_diff_string": list(),
             "sup_diff_string": list(), "opt_auc": list()}
    for param in uniq_param:
        df_param = df[df["param"] == param]
        opt_auc = df_param["opt_auc"].unique()[0]
        if variant == "quantile":
            low_auc_diff = df_param["auc_diff"].quantile(alpha/2)
            low_sup_diff = df_param["sup_diff"].quantile(alpha/2)
            med_auc_diff = df_param["auc_diff"].median()
            med_sup_diff = df_param["sup_diff"].median()
            high_auc_diff = df_param["auc_diff"].quantile(1-alpha/2)
            high_sup_diff = df_param["sup_diff"].quantile(1-alpha/2)
            auc_diff_string = "{:.2f} ({:.2f}, {:.2f})".format(
                med_auc_diff, low_auc_diff, high_auc_diff)
            sup_diff_string = "{:.2f} ({:.2f}, {:.2f})".format(
                med_sup_diff, low_sup_diff, high_sup_diff)
        else:
            mean_auc_diff = df_param["auc_diff"].mean()
            mean_sup_diff = df_param["sup_diff"].mean()
            var_auc_diff = df_param["auc_diff"].std()
            var_sup_diff = df_param["auc_diff"].std()
            auc_diff_string = "{:.2f} (pm {:.2f})".format(mean_auc_diff,
                                                          2*var_auc_diff)
            sup_diff_string = "{:.2f} (pm {:.2f})".format(mean_sup_diff,
                                                          2*var_sup_diff)
        d_res["param"].append("{:.4f}".format(param))
        d_res["opt_auc"].append("{:.2f}".format(opt_auc))
        d_res["auc_diff_string"].append(auc_diff_string)
        d_res["sup_diff_string"].append(sup_diff_string)
    df_res = pd.DataFrame(d_res)
    with open(json_name.split(".")[0] + "_" + variant + ".tex", "wt") as f:
        f.write(df_res.to_latex(index=False))

# ---------- Other tests ----------

def main_plot():
    """Used to test the data generation and ROC calculation by plots."""
    gamma = 0.001
    dim = 2 # must be always pair, since we generate pairs of instances.
    p = 0.5
    gt_depth = 2
    n_samples = 400
    n_train = 10000
    n_test = 10000
    depth = 5

    tree = generate_tree(dim=dim, depth=gt_depth, gamma=gamma, p=p)
    X, z = generate_sample_from_tree(n_samples, tree)
    plt.figure(figsize=(4, 4))
    plt.scatter(X[z == +1, 0], X[z == +1, 1], label="pos", color="green",
                alpha=0.4)
    plt.scatter(X[z == 0, 0], X[z == 0, 1], label="neg", color="red",
                alpha=0.4)
    plt.legend()
    plt.grid()
    plt.savefig("scatter_simtree.pdf")

    print("Done first plot... - " + datetime.now().ctime())
    X_train, z_train = generate_sample_from_tree(n_train, tree)
    X_test, z_test = generate_sample_from_tree(n_test, tree)

    print("Done generating the sample... - " + datetime.now().ctime())

    gen_split = DiagSymSplitter
    model = SimTreeRank(gen_split, depth)
    model.fit_pairs(X_train, z_train)

    print("Done fitting the model... - " + datetime.now().ctime())
    s_test = model.score(X_test)
    print("Done scoring the pairs... - " + datetime.now().ctime())

    plt.figure(figsize=(4, 4))
    plt.plot(*simtree_optimal_roc(tree), label="optimal")
    plt.plot(*roc_curve(simtree_eta(tree, X_test), z_test), label="opt rand")
    plt.plot(*roc_curve(s_test, z_test), label="model")
    plt.plot([0, 1], [0, 1], label="random")
    plt.legend()
    plt.grid()
    plt.savefig("roc_simtree.pdf")
    print("Done plotting... - " + datetime.now().ctime())

    roc_optimal = ptwise_roc(*simtree_optimal_roc(tree))
    roc_model = ptwise_roc(*roc_curve(s_test, z_test))
    sup_diff = np.max(np.abs(np.array(roc_optimal) - np.array(roc_model)))
    print("Norm sup diff = {:.2f} - {}".format(sup_diff,
                                               datetime.now().ctime()))

    auc_model = auc(s_test, z_test)
    auc_diff = tree.auc - auc_model
    print("AUC diff = {:.2f} - {}".format(auc_diff, datetime.now().ctime()))


# Main:
if __name__ == "__main__":
    main_experiments()
    data_analysis("class_asymmetry.json")
    data_analysis("model_bias.json")
    data_analysis("model_complexity.json")
    # main_plot()
