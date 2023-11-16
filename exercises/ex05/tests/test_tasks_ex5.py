from unittest import TestCase

from nlpwdlfw.datasets import SyntheticBinaryClassificationDataset, BinaryClassificationDataset
from nlpwdlfw.layers import LinearLayer, InputLayer, ReLULayer, SoftMaxLayer
from nlpwdlfw.nodes import SigmoidNode, CrossEntropyLoss, InputNode, LinearNode, ReLUNode
from nlpwdlfw.optimizers import OnlineGradientDescent
from nlpwdlfw.utils import graph_to_graphviz, accuracy, predict


class TestTasksEx5(TestCase):

    def test_task1_1(self):
        i1 = InputNode()
        i1.set_value(-2)
        r1 = ReLUNode([i1])
        self.assertAlmostEqual(0.0, r1.value(), 5)

        i2 = InputNode()
        i2.set_value(0)
        r2 = ReLUNode([i2])
        self.assertAlmostEqual(0.0, r2.value(), 5)

        i3 = InputNode()
        i3.set_value(2)
        r3 = ReLUNode([i3])
        self.assertAlmostEqual(2.0, r3.value(), 5)

    def test_task1_2(self):
        i1 = InputNode()
        i1.set_value(-2)
        r1 = ReLUNode([i1])
        self.assertAlmostEqual(0.0, r1.local_partial_derivatives_wrt_children()[0], 5)

        i2 = InputNode()
        i2.set_value(0)
        r2 = ReLUNode([i2])
        self.assertAlmostEqual(1.0, r2.local_partial_derivatives_wrt_children()[0], 5)

        i3 = InputNode()
        i3.set_value(2)
        r3 = ReLUNode([i3])
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
        # two input nodes
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
