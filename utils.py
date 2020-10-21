import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def image_normalizer(imgs):
    """
    Nornalise images
    """
    std = np.std(imgs)
    #std = 128
    mean = np.mean(imgs)
    #mean = 128
    return (imgs - mean) / std


def image_grayscale(imgs, dist): 
    print("toto")
	
    gray = np.sum(dist/3, axis=3, keepdims=True)
	
    return gray
