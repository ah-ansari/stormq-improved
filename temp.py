import numpy as np
import networkx as nx
import tools
import actions
import competition

q_tables = [np.load("5/"+str(x)+"/q_table.npy") for x in range(5)]

for i in range(5):
    G = nx.read_gpickle("graph")
    print(str(i)+"-----------------")
    competition.compete(G, q_tables[i])

