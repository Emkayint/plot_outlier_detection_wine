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

colors = ["m", "g", "b"]
legend1 = {}
legend2 = {}

# get data from wine function dataset

X1 = load_wine()["data"][:, [1, 2]] # two clusters

# Learn a frontier for outlier detection with several classifiers 
xx1, yy1 = np.meshgrid(np.linspace(0, 6, 500), np.linspace(1, 4.5, 500))

for i, (clf_name, clf) in enumerate(classifiers.items()):
  plt.figure(1)
  clf.fit(X1)
  Z1 = clf.decison_function(np.c_[xx1.ravel(), yy1.ravel()])
  Z1 = Z1.reshape(xx1.shape)
  legend1[clf_name] = plt.contour(xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])

legend1_values_list = list(legend1.values())
legend1_keys_list = list(legend1.keys())
