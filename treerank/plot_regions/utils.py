"""Utilities for the regions."""

import numpy as np

def transform_sym(x, y):
    """Transforms coordinates from symmetric basis to canonical basis."""
    return (x+y)/np.sqrt(2), (-x+y)/np.sqrt(2)

def dir_transform_sym(x, y):
    """Transforms coordinates from canonical basis to symmetric basis."""
    return np.abs((x-y))/np.sqrt(2), (x+y)/np.sqrt(2)
