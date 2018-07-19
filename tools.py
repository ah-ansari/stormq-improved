import numpy as np
import networkx as nx
import random


def load_graph_directed(file):
    # loading edges from file
    edge_list = np.loadtxt(file, dtype=int)

    # create the graph
    g = nx.DiGraph()
    g.add_edges_from(edge_list)

    # adding attributes to nodes and edges
    for n in g.nodes():
        # adding threshold to node
        g.add_node(n, t=np.random.random())
        # adding weight to the edge
        in_deg = g.in_degree(n)
        if in_deg != 0:
            g.add_edges_from(g.in_edges(n), w=(1/in_deg))
    # adding activation weights
    g.add_edges_from(g.edges(), w1=0)
    g.add_edges_from(g.edges(), w2=0)
    g.add_edges_from(g.edges(), free_end=1)
    # defining 3 lists to hold free nodes, first player and second player nodes
    g.graph['free'] = g.nodes()
    g.graph['1'] = []
    g.graph['2'] = []

    return g


def load_graph_undirected(file):
    # loading edges from file
    edges = np.loadtxt(file, dtype=int)

    edge_list = np.zeros((2 * edges.shape[0], 2))
    edge_list[0:edges.shape[0], :] = edges[:, :]

    edge_list[edges.shape[0]:, 0] = edges[:, 1]
    edge_list[edges.shape[0]:, 1] = edges[:, 0]

    # create the graph
    g = nx.DiGraph()
    g.add_edges_from(edge_list)

    # adding attributes to nodes and edges
    for n in g.nodes():
        # adding threshold to node
        g.add_node(n, t=np.random.random())
        # adding weight to the edge
        in_deg = g.in_degree(n)
        if in_deg != 0:
            g.add_edges_from(g.in_edges(n), w=(1/in_deg))
    # adding activation weights
    g.add_edges_from(g.edges(), w1=0)
    g.add_edges_from(g.edges(), w2=0)
    g.add_edges_from(g.edges(), free_end=1)
    # defining 3 lists to hold free nodes, first player and second player nodes
    g.graph['free'] = g.nodes()
    g.graph['1'] = []
    g.graph['2'] = []

    return g


def activate_node(g, node, player):
    # First setting the edge weight to be activated
    for u, v, d in g.edges(node, data=True):
        d['w'+str(player)] = d['w']
    # Second adding nodes to lists
    g.graph['free'].remove(node)
    if player == 1:
        g.graph['1'].append(node)
    elif player == 2:
        g.graph['2'].append(node)
    # Third setting edges to be non free(free=0)
    for edge in g.in_edges(node, data=True):
        edge[2]['free_end'] = 0

    return


def diffuse(g):
    activated_first = []
    activated_second = []

    activated_nodes = g.graph['1'] + g.graph['2']
    free_nodes_set = set(g.graph['free'])

    for node_parent in activated_nodes:
        for node in g.neighbors(node_parent):
            if node in free_nodes_set:
                threshold = g.node[node]['t']
                sum1 = g.in_degree(node, 'w1')
                sum2 = g.in_degree(node, 'w2')
                if sum1 > threshold and sum1 > sum2:
                    activated_first.append(node)
                if sum2 > threshold and sum2 > sum1:
                    activated_second.append(node)

    activated_first = np.unique(activated_first)
    activated_second = np.unique(activated_second)
    return activated_first, activated_second


def get_feature(g):
    # 0. Number of free nodes
    # 1. Summation of degrees of all free nodes
    # 2. Summation of weight of the edges for which both vertices are free
    # 3. Maximum degree among all free nodes
    # 4. Maximum sum of free out-edge weight of a node among all nodes
    # 5. Maximum sum of free out-edge weight of a node among nodes which are the first player's neighbors
    # 6. Maximum sum of free out-edge weight of a node among nodes which are the second player's neighbors

    f = np.zeros(7)
    free_nodes = g.graph['free']
    free_nodes_set = set(free_nodes)

    first_player_neighbors = set()
    for node in g.graph['1']:
        first_player_neighbors |= set(g.neighbors(node))
    second_player_neighbors = set()
    for node in g.graph['2']:
        second_player_neighbors |= set(g.neighbors(node))

    # 0
    f[0] = len(free_nodes)
    # 1, 3
    degrees = list(g.out_degree(free_nodes).values())
    if degrees:
        f[1] = sum(degrees)
        f[3] = max(degrees)

    # 2, 4, 5, 6
    f2 = 0
    max4 = 0
    max5 = 0
    max6 = 0

    for node in free_nodes:
        weight_free = 0
        for edge in g.edges(node, data=True):
            if edge[1] in free_nodes_set:
                weight_free += edge[2]['w']
        f2 += weight_free

        if weight_free > max4:
            max4 = weight_free
        if node in first_player_neighbors:
            if weight_free > max5:
                max5 = weight_free
        if node in second_player_neighbors:
            if weight_free > max6:
                max6 = weight_free

    f[2] = f2
    f[4] = max4
    f[5] = max5
    f[6] = max6

    return f


def feature_map_lmh(feature, feature_lmh_ranges):
    result = np.zeros(7)
    for i in range(7):
        if feature[i] < feature_lmh_ranges[i, 0]:
            result[i] = 0
        elif feature_lmh_ranges[i, 0] <= feature[i] < feature_lmh_ranges[i, 1]:
            result[i] = 1
        elif feature[i] >= feature_lmh_ranges[i, 1]:
            result[i] = 2

    return result


def feature_map_number(feature_lmh):
    n = 0
    for i in range(7):
        n += feature_lmh[i]*pow(3, 6-i)

    return int(n)


def get_state(g: nx.Graph):
    r1 = len(g.graph['1'])
    r2 = len(g.graph['2'])
    diff = r1 - r2

    r = 0
    if diff < -100:
        r = 0
    elif -100 <= diff < -50:
        r = 1
    elif -50 <= diff < 0:
        r = 2
    elif 0 <= diff < 50:
        r = 3
    elif 50 <= diff < 100:
        r = 4
    elif 100 <= diff:
        r = 5

    return r


def create_random_graph(g: nx.Graph):
    n1 = random.randint(1, len(g.graph['free']) - 2)
    n2 = random.randint(1, len(g.graph['free']) - n1)
    for i in range(n1):
        r = random.randint(0, len(g.graph['free']) - 1)
        activate_node(g, g.graph['free'][r], 1)
    for i in range(n2):
        r = random.randint(0, len(g.graph['free']) - 1)
        activate_node(g, g.graph['free'][r], 2)
    return g
