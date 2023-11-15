from typing import List
from nlpwdlfw.nodes import ScalarNode, InputNode, LinearNode, ParameterNode


class Layer:

    def __init__(self):
        self.nodes = []


class InputLayer(Layer):

    def __init__(self, dim: int):
        super().__init__()
        # Create "dim" input nodes
        for _ in range(dim):
            self.nodes.append(InputNode())

    def set_values_from_training_example_feature_vector(self, values: List[float]) -> None:
        assert len(values) == len(self.nodes)
        for i in range(len(values)):
            i_th_node = self.nodes[i]
            assert isinstance(i_th_node, InputNode)
            i_th_node.set_value(values[i])


class LinearLayer(Layer):
    def __init__(self, output_dim: int, previous_layer: Layer):
        super().__init__()
        # --- EX4_TASK_1 ---
        for i in range(output_dim):
            # Create a fully connected linear node ("linear neuron")
            previous_layer_nodes = previous_layer.nodes
            # We need the same amount of parameters, initialize to 0.01 (that's good for now)
            weights = []
            for _ in range(len(previous_layer_nodes)):
                weights.append(ParameterNode(0.01))
            bias = ParameterNode(0.01)

            # and add a linear node
            self.nodes.append(LinearNode(previous_layer_nodes, weights, bias))
        # --- EX4_TASK_1 ---
