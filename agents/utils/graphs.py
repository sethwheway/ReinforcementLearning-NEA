import matplotlib.pyplot as plt
import numpy as np


def plot_score_graph(scores, group=100, highest=False):
    groups = ([], [])
    for i in range(group, len(scores) + 1, group):
        groups[0].append(i)
        groups[1].append(np.mean(scores[i - group:i]))

    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(groups[0], groups[1])

    if highest:
        plt.plot(max(scores), scores.index(max(scores)), "x")

    plt.show()
