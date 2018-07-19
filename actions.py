import numpy as np
import networkx as nx
import operator


def action_degree(g: nx.Graph):
    free_nodes = g.graph['free']
    degrees = g.out_degree(free_nodes, weight="free_end")
    node = max(degrees.items(), key=operator.itemgetter(1))[0]
    return node


def action_weight(g: nx.Graph):
    free_nodes = g.graph['free']
    free_nodes_set = set(free_nodes)
    max_node = -1
    max_value = -1
    for node in free_nodes:
        weight_free = 0
        for edge in g.edges(node, data=True):
            if edge[1] in free_nodes_set:
                weight_free += edge[2]['w']
        if weight_free > max_value:
            max_value = weight_free
            max_node = node
    return max_node


def action_blocking(g:  nx.Graph, player):
    free_nodes = g.graph['free']
    free_nodes_set = set(free_nodes)
    opponent = 3 - player
    opponent_nodes = g.graph[str(opponent)]

    if not opponent_nodes:
        return action_weight(g)

    opponent_neighbors = set()
    for node in opponent_nodes:
        opponent_neighbors |= set(g.neighbors(node))

    free_opponent_neighbors = free_nodes_set.intersection(opponent_neighbors)
    free_opponent_neighbors = list(free_opponent_neighbors)

    if not free_opponent_neighbors:
        return action_weight(g)

    max_node = -1
    max_value = -1
    for node in free_opponent_neighbors:
        weight_free = 0
        for edge in g.edges(node, data=True):
            if edge[1] in free_nodes_set:
                weight_free += edge[2]['w']
        if weight_free > max_value:
            max_value = weight_free
            max_node = node

    return max_node


def action_first(g: nx.Graph):
    return g.graph['free'][0]


def action_last(g: nx.Graph):
    return g.graph['free'][-1]


def action_min_degree(g: nx.Graph):
    free_nodes = g.graph['free']
    degrees = g.out_degree(free_nodes, weight="free_end")
    node = min(degrees.items(), key=operator.itemgetter(1))[0]
    return node
