"""
    MNIST experiments for TreeRank for similarity learning.
"""
from datetime import datetime
from itertools import product

from utils import (
    fit_n_save_roc,
    plot_rocs_n_save,
    ModelInterface
)

ALL_MODELS = ("LMNN", "nn_sce", "RF")
ALL_DBS = ["MNIST"]

# logging.basicConfig(filename='test.log',
#                     format='%(levelname)s - %(asctime)s - %(message)s',
#                     level=logging.DEBUG, datefmt='%m/%d/%y %I:%M:%S', #%p',
#                     filemode="w")

def main(model_db_names=product(ALL_DBS, ALL_MODELS)):
    for dbname, model_name in model_db_names:
        print("Fitting {} - ".format(dbname) + datetime.now().ctime())
        print("# For model {} - ".format(model_name)
              + datetime.now().ctime())
        fit_n_save_roc(ModelInterface(model_name), dbname,
                       model_folder="exps/real/models",
                       roc_folder="exps/real/rocs")

if __name__ == "__main__":
    main()
    print("Plotting of ROC curves")
    plot_rocs_n_save(outfolder="exps/real/rocs",
                     dbname="MNIST", model_names=ALL_MODELS,
                     infolder="exps/real/rocs",
                     logscale=False)
