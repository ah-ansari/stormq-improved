import numpy as np
import networkx as nx
import tools
import random
import actions
import run
from tqdm import tqdm


def train(graph):
    alpha = 0.5
    eps = 0.8
    gamma = 0.98
    d = 0.998

    opponent_action = run.opponent

    q_table = np.zeros((6, 5))

    for iteration in tqdm(range(run.train_n_rounds)):
        g = graph.copy()
        state = tools.get_state(g)

        for t in range(140):
            # first player action
            if np.random.random() < eps:
                action_choice1 = np.random.randint(5)
            else:
                action_choice1 = np.argmax(q_table[state, :])

            if action_choice1 == 0:
                seed1 = actions.action_degree(g)
            elif action_choice1 == 1:
                seed1 = actions.action_weight(g)
            elif action_choice1 == 2:
                seed1 = actions.action_blocking(g, 1)
            elif action_choice1 == 3:
                seed1 = actions.action_last(g)
            elif action_choice1 == 4:
                seed1 = actions.action_min_degree(g)

            # illegal action
            if seed1 == -1:
                print("Illegal action")
                action_choice1 = 0
                seed1 = actions.action_degree(g)

            tools.activate_node(g, seed1, 1)

            # second player action
            action_choice2 = opponent_action
            if action_choice2 == 0:
                seed2 = actions.action_degree(g)
            elif action_choice2 == 1:
                seed2 = actions.action_weight(g)
            elif action_choice2 == 2:
                seed2 = actions.action_blocking(g, 2)

            # illegal action
            if seed2 == -1:
                print("Illegal action")
                action_choice2 = 0
                seed2 = actions.action_degree(g)
            tools.activate_node(g, seed2, 2)

            a1, a2 = tools.diffuse(g)
            for n in a1:
                tools.activate_node(g, n, 1)
            for n in a2:
                tools.activate_node(g, n, 2)

            next_state = tools.get_state(g)

            r = 0
            if t % 6 == 0:
                r = len(g.graph['1']) - len(g.graph['2'])

            q_table[state, action_choice1] = (1 - alpha) * q_table[state, action_choice1] + alpha * (r + gamma * max(q_table[next_state, :]))
            state = next_state

        alpha = d*alpha
        eps = 0.9 - (0.6*iteration/500)

    return q_table
