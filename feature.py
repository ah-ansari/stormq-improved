import numpy as np
import networkx as nx
import sklearn.cluster
import tools
import actions
import run


def get_features_values_by_competition(graph):
    iterations = run.feature_iterations
    features = np.zeros((iterations*1600, 7))
    counter = 0
    for iteration in range(iterations):
        g = graph.copy()

        while len(g.graph['free']) > 1:
            feature = tools.get_feature(g)
            features[counter][:] = feature
            counter += 1

            # first player action
            action_choice1 = np.random.randint(3)
            if action_choice1 == 0:
                seed1 = actions.action_degree(g)
            elif action_choice1 == 1:
                seed1 = actions.action_weight(g)
            elif action_choice1 == 2:
                seed1 = actions.action_blocking(g, 1)

            # illegal action
            if seed1 == -1:
                print("Illegal action")
                action_choice1 = 0
                seed1 = actions.action_degree(g)
            tools.activate_node(g, seed1, 1)

            # second player action
            action_choice2 = np.random.randint(3)
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

    return features[0:counter, :]


def get_features_values_by_sampling(graph):
    iterations = run.feature_iterations
    features = np.zeros((iterations, 7))

    for iteration in range(iterations):
        g = graph.copy()
        g = tools.create_random_graph(g)
        features[iteration, :] = tools.get_feature(g)
        # print(iteration)
    return features


def get_ranges(features):
    values = np.zeros((7, 2))
    for i in range(7):
        k_means = sklearn.cluster.KMeans(n_clusters=3, random_state=0).fit(features[:, i].reshape(-1, 1))
        centers = k_means.cluster_centers_
        centers.sort(axis=0)
        values[i, 0] = (centers[0]+centers[1]) / 2
        values[i, 1] = (centers[1] + centers[2]) / 2
    return values
