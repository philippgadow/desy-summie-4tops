import matplotlib.pyplot as plt
import mplhep as hep
import networkx as nx
from torch_geometric.utils import to_networkx


# Plotting helper functions

def visualizeGraph(data, loss=None):
    G = to_networkx(data, to_undirected=True)
    color=data.y
    fig = plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2")
    fig.savefig('test_graph.png')


def visualizeBDTScore(twoclass_output, y, class_names, plot_range, plot_colors, outputname):
    fig, ax = plt.subplots()
    for i, n, c in zip(range(2), class_names, plot_colors):
        plt.hist(twoclass_output[y == i],
                 bins=10,
                 range=plot_range,
                 facecolor=c,
                 label='Class %s' % n,
                 alpha=.5,
                 edgecolor='k')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2 * 1.2))
    plt.legend(loc='upper right')
    plt.ylabel('Samples')
    plt.xlabel('Score')
    plt.title('Decision Scores')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    fig.savefig(outputname)