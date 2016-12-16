# author: Hendrik Werner s4549775
# author: Constantin Blach s4329872

import matplotlib.pyplot as plt
from scipy.io import loadmat

# exercise 1.1
synths = [
    loadmat("./data/synth1.mat")
    , loadmat("./data/synth2.mat")
    , loadmat("./data/synth3.mat")
    , loadmat("./data/synth4.mat")
]

for i, synth in enumerate(synths):
    # scatterplot
    X = synth["X"]
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("Synth {}".format(i + 1))
    plt.xlabel("Attribute 1")
    plt.ylabel("Attribute 2")
    plt.show()

# exercise 1.2
# exercise 1.3
