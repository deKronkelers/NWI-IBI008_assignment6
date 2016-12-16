# author: Hendrik Werner s4549775
# author: Constantin Blach s4329872

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

# exercise 1.1
synths = [
    loadmat("./data/synth1.mat")
    , loadmat("./data/synth2.mat")
    , loadmat("./data/synth3.mat")
    , loadmat("./data/synth4.mat")
]

for i, synth in enumerate(synths, 1):
    # scatterplot
    X = synth["X"]
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("Synth {}".format(i))
    plt.xlabel("Attribute 1")
    plt.ylabel("Attribute 2")
    # plt.savefig("assignment6_1_scatterplot_synth{}.pdf".format(i))
    plt.show()

    # classification
    X_train = synth["X_train"]
    y_train = synth["y_train"]
    X_test = synth["X_test"]
    y_test = synth["y_test"]
    f, axes = plt.subplots(nrows=2, ncols=3)
    f.set_figwidth(18)
    f.set_figheight(18)
    for row, metric in enumerate(["euclidean", "cityblock"]):
        for col, n in enumerate([1, 5, 10]):
            classifier = KNeighborsClassifier(metric=metric, n_neighbors=n)
            classifier.fit(X_train, y_train.ravel())
            confusion = confusion_matrix(y_test, classifier.predict(X_test))
            accuracy = classifier.score(X_test, y_test)
            axes[row][col].imshow(confusion, cmap="Greys")
            axes[row][col].set_title(
                "{n} neighbors, {metric}, accuracy={accuracy}".format(
                    i
                    , metric=metric
                    , n=n
                    , accuracy=accuracy
                )
            )
            axes[row][col].set_xlabel("predicted class")
            axes[row][col].set_ylabel("true class")
    f.suptitle("Confusion Matrices Synth {}".format(i))
    # f.savefig("assignment6_1_confusion_matrices_synth{}.pdf".format(i))
    plt.show()

# exercise 1.2
iris = load_iris(return_X_y=True)
X = iris[0]
y = iris[1]

leave_one_out = LeaveOneOut()
k_range = range(1, 41)
average_errors = []

for k in k_range:
    errors = []
    for train_indices, test_indices in leave_one_out.split(X):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        errors.append(1 - classifier.score(X_test, y_test))
    average_errors.append(np.mean(errors))

plt.plot(k_range, average_errors)
plt.title("Iris K Neighbors cross validation")
plt.xlabel("k")
plt.ylabel("average classification error")
# plt.savefig("assignment6_1_average_classification_error.pdf")
plt.show()

# exercise 1.3
wine = loadmat("./data/wine.mat")
X = wine["X"]

X_without_alc = X[:, :10]
k_range = range(1, 41)
average_errors = []

classifier = NearestNeighbors(n_neighbors=40)
classifier.fit(X_without_alc)
neighbors = X[classifier.kneighbors(return_distance=False)]

for k in k_range:
    neighbors_alc = neighbors[:, :k, 10]
    average_errors.append(mean_squared_error(X[:, 10], neighbors_alc.mean(axis=1)))

plt.plot(k_range, average_errors)
plt.title("Wine KNN regression\nAlcohol Prediction")
plt.xlabel("k")
plt.ylabel("mean squared error")
plt.show()
