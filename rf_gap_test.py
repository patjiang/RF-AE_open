import numpy as np
from rfgap import RFGAP
from sklearn.datasets import load_iris

# Load iris data
data = load_iris()
x = data.data
y = data.target

# RFGAP classification
prediction_type = 'classification'

rf = RFGAP(prediction_type = prediction_type)
rf.fit(x, y)

# Get RFGAP probabilities
proximities = rf.get_proximities()
print(proximities.shape)

# TODO: How to incorporate RFPHATE for the current data and dataprep

# Get a distance matrix for kernel
from sklearn.metrics.pairwise import euclidean_distances
proximities = euclidean_distances(proximities)

# TODO: Max normalize the proximities as described in Section (3.3)

# Use kmedoids for selection
import kmedoids
km = kmedoids.KMedoids(5, method='fasterpam')
c = km.fit_predict(proximities)
print(c)