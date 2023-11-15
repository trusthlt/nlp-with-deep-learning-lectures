from unittest import TestCase

from nlpwdlfw.datasets import SyntheticBinaryClassificationDataset, BinaryClassificationDataset
from nlpwdlfw.layers import LinearLayer, InputLayer
from nlpwdlfw.nodes import SigmoidNode, CrossEntropyLoss, InputNode, LinearNode
from nlpwdlfw.optimizers import OnlineGradientDescent
from nlpwdlfw.utils import graph_to_graphviz, accuracy, predict


class TestTasksEx5(TestCase):

    def test_task1_1(self):
        pass

