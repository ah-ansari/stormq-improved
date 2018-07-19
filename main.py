import networkx as nx
import tools
import train
import competition
import run

run.id = 2
run.data_set = "CA-GrQc.txt"
run.data_set_directed = False
run.train_n_rounds = 700
run.opponent = 0

run.description = "t in range(140)" \
                  "actions: degree-weight-blocking-last-mindegree\n" \
                  "states: -100 -50 0 50 100\n" \
                  "epsilon being tuned\n" \
                  "time is included\n" \
                  "selecting random in unseen states in competition\n" \
                  "starting from start till end\n" \
                  "delayed reward t%6 == 0\n" \
                  "alpha = 0.5 - eps = 0.8 - gamma = 0.98 - d = 0.998   \n" \
                  "eps is tuned:eps = u - ((u-l)*t/max,alpha tuned: alpha=alpha*d\n"

run.save_initial()

if run.data_set_directed:
    G = tools.load_graph_directed("datasets\\"+run.data_set)
else:
    G = tools.load_graph_undirected("datasets\\"+run.data_set)
nx.write_gpickle(G, "graph")

for i in range(run.n_run):
    G = nx.read_gpickle("graph")
    print("run: "+str(i)+"-------------------------")
    q_table = train.train(G)

    # competition
    result = competition.compete(G, q_table)
    run.save_one_run(i, q_table, result)
