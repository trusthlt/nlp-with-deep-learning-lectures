from typing import List
from unittest import TestCase
from nlpwdlfw.nodes import SumNode, ConstantNode, ProductNode, ScalarNode


class TestNodes(TestCase):

    def test_task1(self):
        value = 2
        # --- TASK_1 ---
        
        # --- TASK_1 ---
        self.assertEqual(1, value)

    def test_task2_sum_node1(self):
        # w_1 = 2
        # w_2 = 3
        # y = w_1 + w_2
        w_1 = ConstantNode(2)
        w_2 = ConstantNode(3)
        y = SumNode([w_1, w_2])
        self.assertEqual(5, y.value())

        # gradient is a vector of ones
        local_gradient = y.local_partial_derivatives_wrt_arguments()
        self.assertEqual(local_gradient[0], 1.0)
        self.assertEqual(local_gradient[1], 1.0)

    def test_task3_product_node1(self):
        # w_1 = 2
        # w_2 = 3
        # w_3 = 4
        # y = w_1 * w_2 * w_3
        w_1 = ConstantNode(2)
        w_2 = ConstantNode(3)
        w_3 = ConstantNode(4)
        y = ProductNode([w_1, w_2, w_3])
        self.assertEqual(24, y.value())

        # gradient is [12, 8, 6]
        local_gradient = y.local_partial_derivatives_wrt_arguments()
        self.assertEqual(local_gradient[0], 12.0)
        self.assertEqual(local_gradient[1], 8.0)
        self.assertEqual(local_gradient[2], 6.0)

    def test_task4(self):
        # dummy initialization
        a, b, one, r, s, e = [ConstantNode(1)] * 6
        # --- TASK_4 ---

        # --- TASK_4 ---

        # test that b has two parents
        # little hacky, because of private attributes
        parents: List[ScalarNode] = b.__getattribute__('_parents')
        self.assertEqual(2, len(parents))

        # (2 + 3)(3 + 1) = 20
        self.assertEqual(20, e.value())

    def test_task5(self):
        a = ConstantNode(2)
        b = ConstantNode(3)
        one = ConstantNode(1)
        r = SumNode([a, b])
        s = SumNode([b, one])
        e = ProductNode([r, s])

        self.assertEqual(1.0, e.global_derivative_wrt_self())

        self.assertEqual(4.0, a.global_derivative_wrt_self())
        self.assertEqual(9.0, b.global_derivative_wrt_self())

    def test_task6_1(self):
        a = ConstantNode(1)
        b = ConstantNode(2)

        previous_list = [a, b]
        for _ in range(12):
            h1 = SumNode(previous_list.copy())
            h2 = SumNode(previous_list.copy())
            previous_list = [h1, h2]

        e = ProductNode(previous_list)

        # measure duration of computing the gradient
        import time
        start = time.process_time()

        print(a.global_derivative_wrt_self())
        print(b.global_derivative_wrt_self())

        duration = time.process_time() - start
        print(duration)

        self.assertLess(duration, 2)

    def test_task6_2(self):
        a = ConstantNode(1)
        b = ConstantNode(1)

        previous_list = [a, b]
        for _ in range(500):
            h1 = SumNode(previous_list.copy())
            h2 = SumNode(previous_list.copy())
            previous_list = [h1, h2]

        e = ProductNode(previous_list)

        # measure duration of computing the value
        import time
        start = time.process_time()
        print(e.value())
        duration = time.process_time() - start

        print(duration)

        self.assertLess(duration, 1)
