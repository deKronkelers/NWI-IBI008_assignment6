# author: Hendrik Werner s4549775
# author: Constantin Blach s4329872

import matplotlib.pyplot as plt
import neurolab as nl
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold

# exercise 2.1
xor = loadmat("./data/xor.mat")
X = xor["X"]
y = xor["y"]


def scatter_plot_by_class(X: np.ndarray) -> None:
    X_y0 = X[(y == 0).ravel()]
    X_y1 = X[(y == 1).ravel()]
    plt.scatter(X_y0[:, 0], X_y0[:, 1], color="red", label="Class 0")
    plt.scatter(X_y1[:, 0], X_y1[:, 1], color="blue", label="Class 1")
    plt.xlabel("Attribute 1")
    plt.ylabel("Attribute 2")
    plt.legend(loc="lower left")


scatter_plot_by_class(X)
plt.title("Xor Scatterplot")
# plt.savefig("assignment6_2_scatter_plot_xor.pdf")
plt.show()


# exercise 2.2
def plot_decision_boundary(nw) -> None:
    plt.figure(1)
    plt.hold(True)
    delta = 0.05
    levels = 100
    a = np.arange(-1, 2, delta)
    b = np.arange(-1, 2, delta)
    A, B = np.meshgrid(a, b)
    values = np.zeros(A.shape)

    for i in range(len(a)):
        for j in range(len(b)):
            values[i, j] = nw.sim(np.mat([a[i], b[j]]))[0, 0]
    plt.contour(A, B, values, levels=[.5], colors=['k'], linestyles='dashed')
    plt.contourf(A, B, values, levels=np.linspace(values.min(), values.max(), levels), cmap=plt.cm.RdBu)


def estimate_classification_error(
        X: np.ndarray
        , y: np.ndarray
        , network
        , n_splits: int = 10
) -> int:
    k_fold = KFold(n_splits=n_splits)
    mae = nl.error.MAE()
    errors = []

    for train_indices, test_indices in k_fold.split(X, y):
        network.train(X[train_indices], y[train_indices])
        errors.append(mae(y[test_indices], network.sim(X[test_indices])))

    return np.mean(errors)


def learn_network(
        X: np.ndarray
        , y: np.ndarray
        , hidden_units: int = 1
) -> None:
    network = nl.net.newff(
        [[0, 1], [0, 1]]
        , [hidden_units, 1]
        , [nl.trans.TanSig(), nl.trans.TanSig()]
    )
    print("Estimated Classification error ({} hidden layer): {}".format(
        hidden_units, estimate_classification_error(X, y, network))
    )
    plot_decision_boundary(network)
    scatter_plot_by_class(X)
    plt.title("Neural Network (hidden units={})\nDecision Boundary".format(hidden_units))
    plt.show()


learn_network(X, y)

# exercise 2.3
learn_network(X, y, hidden_units=2)

# exercise 2.4
learn_network(X, y, hidden_units=10)
