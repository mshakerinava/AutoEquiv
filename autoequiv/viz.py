from autoequiv.core import *
from autoequiv.utils import *


def draw_colored_bipartite_graph(colors, node_size=5, edge_width=2):
    import matplotlib.pyplot as plt
    import networkx as nx
    w = dict_to_arr(colors)
    n, m = w.shape
    G = nx.Graph()
    for i in range(n):
        G.add_node('i%d' % i, pos=(i - (n - 1) / 2, +1))
    for i in range(m):
        G.add_node('o%d' % i, pos=(i - (m - 1) / 2, -1))
    for i in range(n):
        for j in range(m):
            G.add_edge('i%d' % i, 'o%d' % j, c=w[i, j])
    pos = nx.get_node_attributes(G, 'pos')
    c_dict = nx.get_edge_attributes(G, 'c')
    c = [c_dict[x] for x in G.edges]
    nx.draw_networkx_edges(G, pos=pos, width=edge_width, edge_color=c, edge_cmap=plt.cm.rainbow, alpha=0.6)
    nx.draw_networkx_nodes(G, pos=pos, node_color='k', node_size=node_size)
    plt.axis('off')


# adapted from code provided by Siamak Ravanbakhsh
def draw_colored_matrix(colors, cmap='rainbow', markersize=20):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx
    import pylab as mpl
    mrklist = ['3', 'd', ',', '2', '8', 's', '|', '^', '<', '4', 'H', '+', 'o', 'v', 'p', 'D', '>', '1', '.', '_', '*', 'h']
    w = dict_to_arr(colors)
    wvals = np.unique(w.ravel()).astype(int)
    cm = plt.get_cmap('Greys') 
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(wvals) - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    if markersize > 0:
        for v, wval in enumerate(wvals):
            x, y = np.nonzero(w == wval)
            x = w.shape[0] - x - 1
            plt.scatter(x, y, s=markersize, marker=mrklist[wval % len(mrklist)], color=scalarMap.to_rgba(wval), alpha=1)
    plt.xticks(np.arange(w.shape[0]-1)+.5, [])
    plt.yticks(np.arange(w.shape[1]-1)+.5, [])
    plt.xlim(-.5, w.shape[0]-.5)
    plt.ylim(-.5, w.shape[1]-.5)
    plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=.5)
    mpl.imshow(w[::-1, :].T, cmap=cmap, alpha=.8, vmin=0, vmax=len(wvals) - 1)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
