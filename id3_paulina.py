import random
import numpy as np
from collections import Counter
import seaborn as sn
import matplotlib.pyplot as plt



class Node:

    def __init__(self, next_nodes={}):
        self.next_nodes = next_nodes
        self.d_id = None

    def set_attr_id(self, d_id):
        self.d_id = d_id

    def set_label(self, label):
        self.label = label

    def set_class_name(self, class_name):
        self.class_name = class_name

    def set_next_nodes(self, next_nodes):
        self.next_nodes = next_nodes



def divide_U(d, U):
    d_values = set([row[d] for row in U])
    U_subsets = {}
    for value in d_values:
        Uj = [row for row in U if row[d] == value]
        U_subsets[value] = Uj
    return U_subsets


def entropy_U(U, class_id):
    classes = [row[class_id] for row in U]
    counter = Counter(classes)
    f = [count/len(U) for count in counter.values()]
    return -sum([fi*np.log(fi) if fi != 0  else 0 for fi in f])


def entropy_subsets(d, U, class_id):
    U_subsets = divide_U(d, U)
    return sum([len(Uj)/len(U)*entropy_U(Uj, class_id) for Uj in U_subsets.values()])


def inf_gain(d, U, class_id):
    I = entropy_U(U, class_id)
    Inf = entropy_subsets(d, U, class_id)
    return I - Inf


def find_the_best_d(U, D, class_id):
    inf_gains = []
    for d in D:
        inf_gains.append(inf_gain(d, U, class_id))
    d_best = max(inf_gains)
    d_best_id = inf_gains.index(d_best)
    return D[d_best_id]


def test(data, first_node, class_id):
    correct = 0
    pred_curr_vals = []
    for row in data:
        val = row[class_id]
        predicted_val = test_current(row, first_node)
        pred_curr_vals.append((predicted_val, val))
        if val == predicted_val:
            correct += 1
    return correct, pred_curr_vals


def test_current(row, current_node):
    if current_node.d_id is None:
        return current_node.class_name
    else:
        d_id = current_node.d_id
        for node_label in current_node.next_nodes:
            node = current_node.next_nodes[node_label]
            if row[d_id] == node_label:
                return test_current(row, node)


def id3(D, U, class_id):
    # D - attributes
    # U - data
    node = Node()
    classes = [row[class_id] for row in U]
    if len(set(classes)) == 1:
        node.class_name = classes[0]
        return node
    if len(D) == 0:
        counter = Counter(classes)
        node.class_name = max(counter, key=counter.get)
        return node
    d_best_id = find_the_best_d(U, D, class_id)
    U_subsets = divide_U(d_best_id, U)  
    node.set_attr_id(d_best_id)
    tree = {}
    D_new = [d for d in D if d!=d_best_id]
    for value, subset in U_subsets.items():
        tree[value] = id3(D_new, subset, class_id)
    node.set_next_nodes(tree)
    return node


def main(path, class_id=0):
    data = []
    with open(path, 'r') as file_handle:
        for line in file_handle:
            data.append(line[:-1].split(","))
    data = [elem[:12] for elem in data]
    D = [i for i in range(len(data[0])) if i != class_id]
    print(D)
    k = len(data)*3//5
    data_valid = random.sample(data, k=k)
    data_test = [x for x in data if x not in data_valid]

    first_node = id3(D, data_valid, class_id)
    correct, pred_curr_vals = test(data_test, first_node, class_id)

    key1, key2 = set([row[class_id] for row in data])
    
    tp = []
    tn = []
    fp = []
    fn = []

    for val in pred_curr_vals:
        if val[0] == val[1] and val[1] == key1:
            tp.append(val)
        elif val[0] == val[1] and val[1] == key2:
            tn.append(val)
        elif val[0] != val[1] and val[1] == key1:
            fp.append(val)
        elif val[0] != val[1] and val[1] == key2:
            fn.append(val)        

    # tp = len([val for val in pred_curr_vals if val[0] == val[1] and val[0] == key1])
    # tn = len([val for val in pred_curr_vals if val[0] == val[1] and val[0] == key2])
    # fp = len([val for val in pred_curr_vals if val[0] != val[1] and val[0] == key1])
    # fn = len([val for val in pred_curr_vals if val[0] != val[1] and val[0] == key2])

    confusion_matrix = [[len(fp), len(tp)], [len(tn), len(fn)]]
    xlabel = [key2, key1]
    ylabel = [key1, key2]

    accuracy = correct/len(data_test)

    plt.figure(figsize = (5, 4))
    plot = sn.heatmap(confusion_matrix, annot=True, fmt="g", xticklabels=xlabel, yticklabels=ylabel)
    plot.set(xlabel="Actual", ylabel="Predicted")
    plt.show()

    print(f'count of data: {len(data)}')
    print(f'count of valid data: {len(data_valid)}')
    print(f'count of test data: {len(data_test)}')
    print(f'count of correct answers: {correct}')
    print(f'accuracy: {accuracy}')

    return correct



path = "/home/paulina/air/wsi/cw4_beta/agaricus-lepiota.data"
# path = "/home/paulina/air/wsi/cw4_beta/breast-cancer.data"
# path = "/home/paulina/air/wsi/cw4_ponownie/elo.data"


if __name__ == "__main__":
    main(path)