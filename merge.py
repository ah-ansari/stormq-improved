import numpy as np
import networkx as nx
import tools
import feature
import train
import competition
import run

G = nx.read_gpickle("graph")
run.opponent = 0

folder = "19/"

q_tables = [np.load(folder + str(x) + "/q_table.npy") for x in range(5)]
ranges = np.load(folder+"0/ranges.npy")


def merge_sum(qs):
    q_table = np.zeros((2187, 3))
    for i in range(5):
        q_table += qs[i]
    return q_table


def merge_avg(qs):
    q_table = np.zeros((2187, 3))
    for i in range(5):
        q_table += qs[i]
    q_table = q_table/len(qs)
    return q_table


def merge_voted(qs):
    q_table = np.zeros((2187, 3))

    for i in range(2187):
        if np.sum(qs[0][i, :]) != 0:
            val = np.zeros((3, 1))
            for x in range(5):
                val[np.argmax(qs[x][i, :])] += 1
            q_table[i, np.argmax(val)] += 10

    return q_table


def merge_weighted(qs, rewards):
    q_table = np.zeros((2187, 3))
    for i in range(5):
        q_table += rewards[i] * qs[i]

    q_table = q_table / np.abs(np.sum(np.abs(rewards)))
    return q_table


rewards = np.array([-147, -152, 220, 215, -167])
# q = merge_sum(q_tables)
# q = merge_voted(q_tables)
q = merge_weighted(q_tables, rewards)

competition.compete(G.copy(), q, ranges)

