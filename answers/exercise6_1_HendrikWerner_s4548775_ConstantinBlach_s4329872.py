# author: Hendrik Werner s4549775
# author: Constantin Blach s4329872

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

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
    plt.show()

# exercise 1.2
# exercise 1.3
