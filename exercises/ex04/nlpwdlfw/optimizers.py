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
                    learning_rate: float = 0.001) -> None:
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
            input_layer.set_values_from_training_example_feature_vector(example.feature_vector)

            # Set the gold label
            gold_label_node.set_value(example.label)

            # --- TODO TASK_2 ---

            # --- TASK_2 ---

            loss_value = loss_node.value()
            print("{:10.4f}".format(loss_value))
