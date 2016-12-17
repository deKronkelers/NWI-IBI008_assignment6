# author: Hendrik Werner s4549775
# author: Constantin Blach s4329872

import neurolab
import matplotlib.pyplot as plt
from scipy.io import loadmat

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
# exercise 2.3
# exercise 2.4
