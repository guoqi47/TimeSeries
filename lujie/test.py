# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 150
random_state = 170
X, y = make_blobs(n_samples=n_samples, n_features=1, random_state=random_state)

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

plt.scatter(X[:, 0], X[:, 0], c=y_pred)

plt.show()