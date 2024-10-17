import math
from typing import List

from nlpwdlfw.nodes import ScalarNode, InputNode, LinearNode, ParameterNode, ReLUNode


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


class ReLULayer(Layer):
    def __init__(self, previous_layer: Layer):
        super().__init__()
        # --- TODO EX5_TASK_2 ---
        # For each previous node
        for previous_node in previous_layer.nodes:
            # Create a new ReLU node
            self.nodes.append(ReLUNode(previous_node))
        # --- EX5_TASK_2 ---


class SoftMaxLayerNode(ScalarNode):

    def __init__(self, previous_layer: Layer, softmax_layer: 'SoftMaxLayer',
                 index_of_self_in_softmax_layer: int) -> None:
        super().__init__(previous_layer.nodes)
        # we need to know to which layer this node belongs to because of the global computation
        # of the denominator
        self._softmax_layer = softmax_layer
        # and also on which position in the softmax layer this node is
        self._index_of_self_in_softmax_layer = index_of_self_in_softmax_layer

    def get_argument_value(self) -> float:
        return self._children[self._index_of_self_in_softmax_layer].value()

    def value(self) -> float:
        if self._cache.value is not None:
            return self._cache.value

        result = 0.0
        # --- TODO EX5_TASK_3 ---
        argument_value = self.get_argument_value()
        denominator = self._softmax_layer.compute_denominator()

        result = math.exp(argument_value) / denominator
        # --- EX5_TASK_3 ---

        # Save to the cache
        self._cache.value = result

        return result

    def local_partial_derivatives_wrt_children(self) -> List[float]:
        if self._cache.local_partial_derivatives_wrt_children is not None:
            return self._cache.local_partial_derivatives_wrt_children

        # --- TODO EX5_TASK_3 ---
        result = []
        for j, child_node in enumerate(self._children):
            if self._index_of_self_in_softmax_layer == j:
                result.append(self.value() * (1 - self.value()))
            else:
                # j-th node value
                j_th_node_value = self._softmax_layer.nodes[j].value()
                result.append(- self.value() * j_th_node_value)
        # --- EX5_TASK_2 ---

        # Save to the cache
        self._cache.local_partial_derivatives_wrt_children = result

        return result


class SoftMaxLayer(Layer):

    def __init__(self, previous_layer: Layer):
        super().__init__()
        # For each previous node
        for i, previous_node in enumerate(previous_layer.nodes):
            # Create a new SoftMax node
            self.nodes.append(SoftMaxLayerNode(previous_layer, self, i))

    def compute_denominator(self) -> float:
        # Naive implementation will result in overflows
        # Also: We have no mechanism for caching on the Layer level (only for each ScalarNode), so this will
        # be very prohibitive operation (repeated many times)
        result = 0.0
        for node in self.nodes:
            assert isinstance(node, SoftMaxLayerNode)
            result += math.exp(node.get_argument_value())
        return result
