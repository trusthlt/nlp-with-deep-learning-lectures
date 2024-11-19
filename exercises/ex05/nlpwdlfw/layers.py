import math
from typing import List, Any

from nlpwdlfw.nodes import ScalarNode, InputNode, LinearNode, ParameterNode, ReLUNode


class Layer:

    def __init__(self):
        self.nodes: List[Any[ScalarNode, InputNode]] = []


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
        # --- EX5_TASK_1 ---

        # --- EX5_TASK_1 ---


class ReLULayer(Layer):
    def __init__(self, previous_layer: Layer):
        super().__init__()
        # --- TODO EX5_TASK_4 ---

        # --- EX5_TASK_4 ---


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
        # --- TODO EX5_TASK_5 ---

        # --- EX5_TASK_5 ---

        # Save to the cache
        self._cache.value = result

        return result

    def local_partial_derivatives_wrt_children(self) -> List[float]:
        if self._cache.local_partial_derivatives_wrt_children is not None:
            return self._cache.local_partial_derivatives_wrt_children

        result = []
        # --- TODO EX5_TASK_5 ---
        pass
        # --- EX5_TASK_5 ---

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
