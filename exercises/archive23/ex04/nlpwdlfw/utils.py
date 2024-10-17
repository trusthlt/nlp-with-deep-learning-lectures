from typing import List

from nlpwdlfw.datasets import BinaryClassificationExample
from nlpwdlfw.layers import InputLayer
from nlpwdlfw.nodes import ScalarNode


def graph_to_graphviz(output_node: ScalarNode):
    buffer = ["digraph G {", "rankdir=LR", "node [shape=box style=filled]"]
    __graph_to_graphviz_recursive(output_node, buffer)
    buffer.append("}")
    print("\n".join(buffer))


def __graph_to_graphviz_recursive(node: ScalarNode, buffer: List[str]):
    for child in node.__getattribute__('_children'):
        buffer.append('"' + repr(child) + '" -> "' + repr(node) + '"')
        __graph_to_graphviz_recursive(child, buffer)


def predict(input_layer: InputLayer, output_node: ScalarNode,
            test_data: List[BinaryClassificationExample]) -> List[float]:
    result = []
    for test_example in test_data:
        # We must clean the cache before any computations!
        output_node.clean_cache_recursively()

        # Set the input feature vector values
        input_layer.set_values_from_training_example_feature_vector(test_example.feature_vector)
        output_node.clean_cache_recursively()
        prediction = output_node.value()
        # Turn probability into 0 or 1
        result.append(1.0 if prediction >= 0.5 else 0.0)

    return result


def accuracy(predictions: List[float], test_data: List[BinaryClassificationExample]) -> float:
    total = 0
    correct = 0
    for i in range(len(predictions)):
        total += 1
        if predictions[i] == test_data[i].label:
            correct += 1
    return correct / total
