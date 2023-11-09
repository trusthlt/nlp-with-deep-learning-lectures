from typing import List
from nlpwdlfw.nodes import ScalarNode, InputNode, LinearNode, ParameterNode


class Layer:
    nodes = []


class InputLayer(Layer):

    def __init__(self, dim: int):
        # Create "dim" input nodes
        self.nodes: List[InputNode] = [InputNode()] * dim

    def set_values_from_training_example_feature_vector(self, values: List[float]) -> None:
        assert len(values) == len(self.nodes)
        for i in range(len(values)):
            self.nodes[i].set_value(values[i])


class LinearLayer(Layer):
    def __init__(self, output_dim: int, previous_layer: Layer):
        pass
        # --- TODO TASK_1 ---

        # --- TASK_1 ---
