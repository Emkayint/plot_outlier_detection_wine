# Author: Martin Kariuki <>
# License: MIT
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.datasets import load_wine

# Define classifiers
classifiers = {
  "Empirical Covariance": EllipticEnvelope(support_fraction=1.0, contamination=0.25),  "Robust Covariance (Minimum Covariance Determinant)" : EllipticEnvelope(contamination=0.25), "OCSVM": OneClassSVM(nu=0.25, gamma=0.35)
}

