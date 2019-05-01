"""DEBUG, temporary"""
from shutil import copyfile
import dill as pickle
from treerank.nn_model import NeuralModel
from treerank.pickable_rf import PickableRandomForest

def save_model(filename, model):
    """Saves a model."""
    if isinstance(model, NeuralModel):
        model.encode_layer.save(filename)
    elif isinstance(model, PickableRandomForest):
        copyfile("treerank/params.json", filename + "_params.json")
        with open(filename, "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    """Loads a saved model."""
    with open(filename, "rb") as f:
        return pickle.load(f)
