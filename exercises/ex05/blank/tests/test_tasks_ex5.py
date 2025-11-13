from unittest import TestCase

from nlpwdlfw.datasets import SyntheticBinaryClassificationDataset, BinaryClassificationDataset
from nlpwdlfw.layers import LinearLayer, InputLayer
from nlpwdlfw.layers import ReLULayer, SoftMaxLayer, SoftMaxLayerNode
from nlpwdlfw.nodes import ReLUNode
from nlpwdlfw.nodes import SigmoidNode, CrossEntropyLoss, InputNode, LinearNode
from nlpwdlfw.optimizers import OnlineGradientDescent
from nlpwdlfw.utils import graph_to_graphviz, accuracy, predict


class TestTasksEx51(TestCase):

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

        self.assertTrue(linear_layer.__getattribute__('nodes')[0].__getattribute__('_weights')[0] is not
                        linear_layer.__getattribute__('nodes')[0].__getattribute__('_weights')[1])

    def test_task1_5(self):
        input_layer = InputLayer(3)
        linear_layer1 = LinearLayer(2, input_layer)
        linear_layer2 = LinearLayer(1, linear_layer1)
        sigmoid_node = SigmoidNode(linear_layer2.nodes[0])

        graph_to_graphviz(sigmoid_node)

        self.assertEqual(1, len(linear_layer2.nodes))

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

        graph_to_graphviz(loss_node)

        OnlineGradientDescent.perform_sgd(10000, loss_node, input_layer, gold_label_input_node,
                                          dataset.get_training_examples())

        # evaluate
        test_examples = dataset.get_test_examples()
        predictions = predict(input_layer, sigmoid_node, test_examples)
        acc = accuracy(predictions, test_examples)
        # We must learn almost 100% accuracy
        self.assertGreater(acc, 0.98)

        # Inspect parameters
        linear_node: LinearNode = linear_layer.__getattribute__('nodes')[0]
        print(linear_node.__getattribute__('_weights')[0].value())
        print(linear_node.__getattribute__('_weights')[1].value())
        print(linear_node.__getattribute__('_bias').value())


class TestTasksEx52(TestCase):

    def test_task1_1(self):
        i1 = InputNode()
        i1.set_value(-2)
        r1 = ReLUNode(i1)
        self.assertAlmostEqual(0.0, r1.value(), 5)

        i2 = InputNode()
        i2.set_value(0)
        r2 = ReLUNode(i2)
        self.assertAlmostEqual(0.0, r2.value(), 5)

        i3 = InputNode()
        i3.set_value(2)
        r3 = ReLUNode(i3)
        self.assertAlmostEqual(2.0, r3.value(), 5)

    def test_task1_2(self):
        i1 = InputNode()
        i1.set_value(-2)
        r1 = ReLUNode(i1)
        self.assertAlmostEqual(0.0, r1.local_partial_derivatives_wrt_children()[0], 5)

        i2 = InputNode()
        i2.set_value(0)
        r2 = ReLUNode(i2)
        # self.assertAlmostEqual(1.0, r2.local_partial_derivatives_wrt_children()[0], 5)
        # value has to be between 0 and 1 for subderivative property
        # see https://en.wikipedia.org/wiki/Subderivative
        self.assertGreaterEqual(r2.local_partial_derivatives_wrt_children()[0], 0.0)
        self.assertLessEqual(r2.local_partial_derivatives_wrt_children()[0], 1.0)

        i3 = InputNode()
        i3.set_value(2)
        r3 = ReLUNode(i3)
        self.assertAlmostEqual(1.0, r3.local_partial_derivatives_wrt_children()[0], 5)

    def test_task2_1(self):
        # two input nodes
        i1 = InputNode()
        i1.set_value(-5)
        i2 = InputNode()
        i2.set_value(5)

        # input layer with these two nodes
        input_layer = InputLayer(2)
        input_layer.nodes = [i1, i2]

        # simply directly into a ReLU layer
        relu_layer = ReLULayer(input_layer)

        # must have two nodes
        self.assertEqual(2, len(relu_layer.nodes))

        # which are instances of ReLU node
        self.assertTrue(isinstance(relu_layer.nodes[0], ReLUNode))
        self.assertTrue(isinstance(relu_layer.nodes[1], ReLUNode))

        # with correct values
        self.assertEqual(relu_layer.nodes[0].value(), 0.0)
        self.assertEqual(relu_layer.nodes[1].value(), 5.0)

        # and derivatives
        # check first there's only a single derivative of each node (sanity check)
        self.assertEqual(len(relu_layer.nodes[0].local_partial_derivatives_wrt_children()), 1)
        self.assertEqual(len(relu_layer.nodes[1].local_partial_derivatives_wrt_children()), 1)
        # and then their values
        self.assertEqual(relu_layer.nodes[0].local_partial_derivatives_wrt_children()[0], 0.0)
        self.assertEqual(relu_layer.nodes[1].local_partial_derivatives_wrt_children()[0], 1.0)

    def test_task3_1(self):
        # three input nodes
        i1 = InputNode()
        i1.set_value(0)
        i2 = InputNode()
        i2.set_value(0)
        i3 = InputNode()
        i3.set_value(1)

        # input layer with these two nodes
        input_layer = InputLayer(3)
        input_layer.nodes = [i1, i2, i3]

        softmax_layer = SoftMaxLayer(input_layer)
        # e / (1 + 1 + e) = 0.576
        # 1 / (1 + 1 + e) = 0.2119
        self.assertAlmostEqual(0.2119, softmax_layer.nodes[0].value(), 2)
        self.assertAlmostEqual(0.2119, softmax_layer.nodes[1].value(), 2)
        self.assertAlmostEqual(0.576, softmax_layer.nodes[2].value(), 2)

    def test_task3_2(self):
        # three input nodes
        i1 = InputNode()
        i1.set_value(0)
        i2 = InputNode()
        i2.set_value(0)
        i3 = InputNode()
        i3.set_value(1)

        # input layer with these two nodes
        input_layer = InputLayer(3)
        input_layer.nodes = [i1, i2, i3]

        softmax_layer = SoftMaxLayer(input_layer)

        first_node = softmax_layer.nodes[0]
        assert isinstance(first_node, SoftMaxLayerNode)
        first_node_derivatives_wrt_children = first_node.local_partial_derivatives_wrt_children()
        # there must be three values
        self.assertEqual(3, len(first_node_derivatives_wrt_children))

        # e / (1 + 1 + e) = 0.576
        # 1 / (1 + 1 + e) = 0.2119

        # 0.2119 ( 1 - 0.2119) = 0.1669
        # - 0.2119 * 0.2119 = - 0.0449
        # - 0.2119 * 0.576 = -0.1221

        self.assertAlmostEqual(0.1669, first_node_derivatives_wrt_children[0], 2)
        self.assertAlmostEqual(-0.0449, first_node_derivatives_wrt_children[1], 2)
        self.assertAlmostEqual(-0.1221, first_node_derivatives_wrt_children[2], 2)
