import numpy as np

# parameters
n = 2
num_hidden_layers = 2
m = [2, 2]
num_nodes_output = 1


def initialize_network(
    num_inputs, num_hidden_layersUsed, num_nodes_hidden, num_nodes_output
):
    num_nodes_previous = num_inputs  # number of nodes in the previous layer
    network = {}

    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layersUsed + 1):

        if layer == num_hidden_layersUsed:
            layer_name = "output"  # last layer = output
            num_nodes = num_nodes_output
        else:
            layer_name = f"layer_{layer+1}"  # hidden layers
            num_nodes = num_nodes_hidden[layer]

        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = f"node_{node+1}"
            network[layer_name][node_name] = {
                "weights": np.around(
                    np.random.uniform(size=num_nodes_previous), decimals=2
                ),
                "bias": np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes  # update for next layer

    return network


def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias


def Node_activation_function(z):
    return 1 / (1 + np.exp(-1 * z))


def forward_propogate(network, inputs):
    layer_inputs = list(inputs)
    for layer in network:
        layer_daya = network[layer]
        layer_outputs = []

        for layer_node in layer_data:

            node_data = layer_data[layer_node]

            node_output = node_activation


# build network
np.random.seed(12)  # fix seed for reproducibility
small_network = initialize_network(5, 3, [3, 2, 3], 1)

# sample inputs
inputs = np.around(np.random.uniform(size=5), decimals=2)

# get weights & bias of first node in first hidden layer
weights = small_network["layer_1"]["node_1"]["weights"]
bias = small_network["layer_1"]["node_1"]["bias"]

# compute weighted sum
z = compute_weighted_sum(inputs, weights, bias)
a = Node_activation_function(z)

print("The output z value: {}".format(z))
