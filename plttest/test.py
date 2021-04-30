
import numpy as np
import matplotlib.pyplot as plt

import copy
from neupy import algorithms, utils

def draw_image(graph, show=True):
    for node_1, node_2 in graph.edges:
        weights = np.concatenate([node_1.weight, node_2.weight])
        line, = plt.plot(*weights.T, color='black')
        plt.setp(line, linewidth=0.2, color='black')

    plt.xticks([], [])
    plt.yticks([], [])
    
    if show:
       	plt.show()

def create_gng(max_nodes, step=0.2, n_start_nodes=2, max_edge_age=50):
    return algorithms.GrowingNeuralGas(
        			n_inputs=2,
        			n_start_nodes=n_start_nodes,

        			shuffle_data=True,
        			verbose=True,

        			step=step,
        			neighbour_step=0.005,

        			max_edge_age=max_edge_age,
        			max_nodes=max_nodes,

        			n_iter_before_neuron_added=100,
        			after_split_error_decay_rate=0.5,
        			error_decay_rate=0.995,
        			min_distance_for_update=0.01,
    		)

def extract_subgraphs(graph):
    subgraphs = []
    edges_per_node = copy.deepcopy(graph.edges_per_node)
    
    while edges_per_node:
        nodes_left = list(edges_per_node.keys())
        nodes_to_check = [nodes_left[0]]
        subgraph = []
        
        while nodes_to_check:
            node = nodes_to_check.pop()
            subgraph.append(node)
    
            if node in edges_per_node:
                nodes_to_check.extend(edges_per_node[node])
                del edges_per_node[node]
                
        subgraphs.append(subgraph)
        
    return subgraphs




from sklearn.datasets import make_moons
data, _ = make_moons(10000, noise=0.06, random_state=0)
plt.scatter(*data.T)
plt.show()



#utils.reproducible()
gng = create_gng(max_nodes=50)


gng.train(data, epochs=10)

draw_image(gng.graph)    
print("Found {} clusters".format(len(extract_subgraphs(gng.graph))))





















