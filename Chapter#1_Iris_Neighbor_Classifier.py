#!/usr/bin/env python
# coding: utf-8

import mglearn
import numpy as np
import pandas as pd
import pandas.plotting as pdp
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
print("Iris Dataset Keys: \n{}".format(iris_dataset.keys()))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print('Data: ')
print('X_train: \n{}'.format(X_train))
print('X_test: \n{}'.format(X_test))
print('y_train: \n{}'.format(y_train))
print('y_test: \n{}'.format(y_test))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pdp.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Prediction of new iris: {}".format(prediction))
print("New iris predicted target name: {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set Score: {:.2f}".format(np.mean(y_pred == y_test)))
#print("Test set Score: {:.2f}".format(knn.score(X_test, y_test)))

