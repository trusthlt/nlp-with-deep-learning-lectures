from unittest import TestCase

from nlpwdlfw.datasets import SyntheticBinaryClassificationDataset, BinaryClassificationDataset
from nlpwdlfw.layers import LinearLayer, InputLayer
from nlpwdlfw.nodes import SigmoidNode, CrossEntropyLoss, InputNode, LinearNode
from nlpwdlfw.optimizers import OnlineGradientDescent


class TestTasksEx4(TestCase):

    def test_synthetic_data(self):
        dataset = SyntheticBinaryClassificationDataset()
        self.assertEqual(len(dataset.get_training_examples()), 200)
        self.assertEqual(len(dataset.get_test_examples()), 20)

    def test_task1_1(self):
        input_layer = InputLayer(4)
        linear_layer = LinearLayer(1, input_layer)

        self.assertEqual(1, len(linear_layer.nodes))

    def test_task1_2(self):
        input_layer = InputLayer(4)
        linear_layer = LinearLayer(2, input_layer)

        self.assertEqual(2, len(linear_layer.nodes))

    def test_task1_3(self):
        input_layer = InputLayer(2)
        linear_layer = LinearLayer(1, input_layer)

        # create fake input feature vector
        input_layer.nodes[0].set_value(100)
        input_layer.nodes[1].set_value(100)

        # Given the current value of the parameters, the output value must be as follows:
        # 100 * 0.01 + 100 * 0.01 + 1 * 0.01 = 2.01
        self.assertAlmostEqual(2.01, linear_layer.nodes[0].value(), 4)

    def test_task1_4(self):
        input_layer = InputLayer(2)
        linear_layer = LinearLayer(2, input_layer)

        weights_n1 = linear_layer.__getattribute__('nodes')[0].__getattribute__('_weights')[0]
        weights_n2 = linear_layer.__getattribute__('nodes')[1].__getattribute__('_weights')[0]
        self.assertTrue(weights_n1 is not weights_n2)

    def test_task2_1(self):
        dataset: BinaryClassificationDataset = SyntheticBinaryClassificationDataset()

        # Our dataset has two features
        input_layer = InputLayer(2)
        # and one label
        gold_label_input_node = InputNode()

        # One linear layer with a scalar output
        linear_layer = LinearLayer(1, input_layer)
        # Add a sigmoid node
        sigmoid_node = SigmoidNode(linear_layer.nodes[0])

        # Cross-entropy loss
        loss_node = CrossEntropyLoss(sigmoid_node, gold_label_input_node)

        OnlineGradientDescent.perform_sgd(1000, loss_node, input_layer, gold_label_input_node,
                                          dataset.get_training_examples())

        # Inspect parameters
        linear_node: LinearNode = linear_layer.__getattribute__('nodes')[0]
        print(linear_node.__getattribute__('_weights')[0].value())
        print(linear_node.__getattribute__('_weights')[1].value())
        print(linear_node.__getattribute__('_bias').value())
