from autoequiv.core import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_colored_bipartite_graph(n, m, colors, edge_width=2.0):
    G = nx.Graph()
    for i in range(n):
        G.add_node('i%d' % i, pos=(i - (n - 1) / 2, +1))
    for i in range(m):
        G.add_node('o%d' % i, pos=(i - (m - 1) / 2, -1))
    for i in range(n):
        for j in range(m):
            G.add_edge('i%d' % i, 'o%d' % j, c=colors[(i, j)])
    fig = plt.figure(figsize=(11, 4))
    pos = nx.get_node_attributes(G, 'pos')
    c_dict = nx.get_edge_attributes(G, 'c')
    c = [c_dict[x] for x in G.edges]
    nx.draw_networkx_edges(G, pos=pos, width=edge_width, edge_color=c, edge_cmap=plt.cm.viridis, alpha=0.6)
    nx.draw_networkx_nodes(G, pos=pos, node_color='k', node_size=5)
    plt.axis('off')
    fig.savefig('bipartite_graph_plot.svg', bbox_inches='tight')
    fig.savefig('bipartite_graph_plot.pdf', bbox_inches='tight')
    fig.savefig('bipartite_graph_plot.png', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def draw_colored_matrix(n, m, colors):
    x = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            x[i, j] = colors.get((i, j), -1)
    ax = plt.matshow(x, cmap='jet')
    fig = ax.figure
    plt.axis('off')
    fig.savefig('matrix_plot.svg', bbox_inches='tight')
    fig.savefig('matrix_plot.pdf', bbox_inches='tight')
    fig.savefig('matrix_plot.png', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def draw_colored_vector(n, colors):
    x = np.zeros((1, n), dtype=int)
    for i in range(n):
        x[0, i] = colors.get(i, -1)
    ax = plt.matshow(x, cmap='jet')
    fig = ax.figure
    plt.axis('off')
    fig.savefig('vector_plot.svg', bbox_inches='tight')
    fig.savefig('vector_plot.pdf', bbox_inches='tight')
    fig.savefig('vector_plot.png', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()
