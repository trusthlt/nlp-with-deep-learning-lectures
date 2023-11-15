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
        # --- TODO EX4_TASK_1 ---

        # --- EX4_TASK_1 ---
