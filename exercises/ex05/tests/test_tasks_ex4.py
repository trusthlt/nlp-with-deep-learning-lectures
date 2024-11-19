import math
from unittest import TestCase

from nlpwdlfw.nodes import ParameterNode, LinearNode, ConstantNode, SigmoidNode, CrossEntropyLoss


class TestTasksEx3(TestCase):

    def test_task0(self):
        pn = ParameterNode(1)
        pn.set_value(2)
        self.assertEqual(pn.value(), 2)

    def test_task1(self):
        # Linear function of two arguments x_1, x_2, two weights and a bias
        # y = 10 * 1 + 11 * 2 - 100 (= -68)
        x_1 = ConstantNode(1)
        x_2 = ConstantNode(2)

        w_1 = ParameterNode(10)
        w_2 = ParameterNode(11)
        b = ParameterNode(-100)

        linear_node = LinearNode([x_1, x_2], [w_1, w_2], b)

        self.assertEqual(linear_node.value(), -68)

    def test_task2(self):
        # Linear function of two arguments x_1, x_2, two weights and a bias
        # y = 10 * 2 - 11
        x_1 = ConstantNode(2)
        w_1 = ParameterNode(10)
        b = ParameterNode(-11)

        linear_node = LinearNode([x_1], [w_1], b)

        self.assertEqual(w_1.global_derivative_wrt_self(), 2)
        self.assertEqual(b.global_derivative_wrt_self(), 1)

    def test_task3(self):
        # Linear function of two arguments x_1, x_2, two weights and a bias
        # y = 10 * 2 - 11
        x_1 = ConstantNode(2)
        w_1 = ParameterNode(10)
        b = ParameterNode(-11)
        linear_node = LinearNode([x_1], [w_1], b)

        s1 = SigmoidNode(linear_node)

        # value must be between 0.0 and 1.0
        self.assertTrue(0.0 < s1.value() < 1.0)

        # Exact value: sigmoid(ln(1)) = 1/2
        ln_of_1 = ConstantNode(math.log(1))
        s2 = SigmoidNode(ln_of_1)
        self.assertAlmostEqual(0.5, s2.value(), 5)

        # Derivative at 0.0 = 0.25
        self.assertAlmostEqual(0.5 * (1 - 0.5), s2.local_partial_derivatives_wrt_children()[0], 5)

    def test_task4_1(self):
        # Linear function of one feature x_1, zero bias, pre-defined w_1 = 1.0
        # y = 1 * 1 + 0
        x_1 = ConstantNode(1)
        w_1 = ParameterNode(1)
        b = ParameterNode(0)
        linear_node = LinearNode([x_1], [w_1], b)
        s1 = SigmoidNode(linear_node)

        # gold label
        y_1 = ConstantNode(1)

        # loss
        loss = CrossEntropyLoss(s1, y_1)

        # This will have small non-negative loss
        self.assertGreater(loss.value(), 0)

    def test_task4_3(self):
        # Derivative
        y_hat = ConstantNode(0.5)
        # gold label
        y = ConstantNode(1)
        # loss
        loss = CrossEntropyLoss(y_hat, y)

        # See also
        # https://www.desmos.com/calculator/uoimlu1enj
        self.assertAlmostEqual(-2.0, loss.local_partial_derivatives_wrt_children()[0], 5)

    def test_task5(self):
        # Linear function of one feature x_1, zero bias, pre-defined w_1 = 1.0
        # y = 1 * 1 + 0
        x_1 = ConstantNode(1)
        w_1 = ParameterNode(1)
        b = ParameterNode(0)
        linear_node = LinearNode([x_1], [w_1], b)
        s1 = SigmoidNode(linear_node)

        # gold label
        y_1 = ConstantNode(1)

        # loss
        loss = CrossEntropyLoss(s1, y_1)

        # This will have small non-negative loss
        previous_loss = loss.value()

        loss.update_parameters_by_gradient_step(learning_rate=0.1)
        loss.clean_cache_recursively()

        new_loss = loss.value()

        # we must end up with a lower loss
        self.assertLess(new_loss, previous_loss)
