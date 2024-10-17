from typing import List

from nlpwdlfw.datasets import BinaryClassificationExample
from nlpwdlfw.layers import InputLayer
from nlpwdlfw.nodes import ScalarNode, InputNode


class OnlineGradientDescent:

    @classmethod
    def perform_sgd(cls, steps: int,
                    loss_node: ScalarNode,
                    input_layer: InputLayer,
                    gold_label_node: InputNode,
                    training_data: List[BinaryClassificationExample],
                    learning_rate: float = 0.01) -> None:
        # we will just iterate over training data
        data_iterator = training_data.__iter__()

        for n in range(steps):
            # Get the next training example if available, otherwise restart from the beginning
            try:
                example = data_iterator.__next__()
            except StopIteration:
                data_iterator = training_data.__iter__()
                example = data_iterator.__next__()

            # Set the input feature vector values
            loss_node.clean_cache_recursively()
            input_layer.set_values_from_training_example_feature_vector(example.feature_vector)

            # Set the gold label
            gold_label_node.set_value(example.label)

            # --- TODO EX4_TASK_2 ---

            # --- EX4_TASK_2 ---
