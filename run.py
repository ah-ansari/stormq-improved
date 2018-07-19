import numpy as np
import os

id = None
data_set = None
data_set_directed = None
train_n_rounds = 500
opponent = None

n_run = 5

description = ""


def save_initial():
    folder = str(id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open(str(id) + "//" + "run_param", "w")
    f.write("data_set: " + data_set + "\n")
    f.write("data_set_directed: " + str(data_set_directed) + "\n")
    f.write("train_n_rounds: " + str(train_n_rounds) + "\n")
    f.write("opponent: " + str(opponent) + "\n")
    f.write("description: \n")
    f.write(description)
    f.close()

    return


def save_one_run(part, q_table, result):
    folder = str(id) + "//" + str(part)
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.savetxt(folder + "//q_table.txt", q_table)
    np.save(folder + "//q_table", q_table)

    f = open(str(id) + "//" + "result", "a")
    f.write("run "+str(part)+": --------------------\n")
    f.write("counter: "+str(result["counter"]) + "\n")
    f.write("unseen: " + str(result["un_seen"]) + "\n")
    f.write("score: " + str(result["score"]) + "\n")
    f.close()

    return
