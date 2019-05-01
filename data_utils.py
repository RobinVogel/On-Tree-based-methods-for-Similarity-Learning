"""All of the data loading utilities."""
import numpy as np

from sklearn.decomposition import PCA

MNIST_FOLDER = "."

def load_preprocess_data(dbname="MNIST"):
    """Loads and preprocesses the data."""
    if dbname == "MNIST":
        return load_preprocess_MNIST()
    return None

# ---------- Database loading

def load_preprocess_MNIST():
    """Returns train_img, train_lab, test_img, test_lab reduced by PCA."""
    X_train, y_train, X_test, y_test = load_MNIST(MNIST_FOLDER
                                                  + "/MNIST_npy_format")

    # Perform PCA to keep 95 percent of total variance.
    pca = PCA(n_components=153) # # Keep 95% of inertia
    pca.fit(X_train)

    X_train_red = pca.fit_transform(X_train)
    X_test_red = pca.transform(X_test)
    return X_train_red, y_train, X_test_red, y_test

def load_MNIST(fold="MNIST_npy_format"):
    """Returns train_img, train_lab, test_img, test_lab."""
    def get_file(x):
        res = np.load(open("{}/{}.npy".format(fold, x), "rb"))
        if x.endswith("_img"):
            res = res.astype(float)/255
        if x.endswith("_lab"):
            res = np.where(res)[1]
        return res
    return (get_file(x) for x in ["train_img", "train_lab",
                                  "test_img", "test_lab"])
