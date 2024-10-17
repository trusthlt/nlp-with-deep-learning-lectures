import unittest
from confusionmatrix import ConfusionMatrix


class Tests(unittest.TestCase):

    def setUp(self):
        self.cm = ConfusionMatrix(["NN", "VB", "ADJ"])
        self.cm.increase_value("NN", "NN", 25)
        self.cm.increase_value("NN", "VB", 5)
        self.cm.increase_value("NN", "ADJ", 1)

        self.cm.increase_value("VB", "NN", 2)
        self.cm.increase_value("VB", "VB", 15)
        self.cm.increase_value("VB", "ADJ", 12)

        self.cm.increase_value("ADJ", "NN", 1)
        self.cm.increase_value("ADJ", "VB", 6)
        self.cm.increase_value("ADJ", "ADJ", 0)
        print(self.cm.get_matrix_copy())

    def test_accuracy(self):
        self.assertAlmostEqual(self.cm.accuracy(), 40 / 67, 2)

    def test_precision_NN(self):
        self.assertAlmostEqual(self.cm.precision("NN"), 25 / 28, 2)

    def test_precision_all(self):
        result = self.cm.precision_all_classes()
        print(result)

    def test_f1_macro(self):
        print(self.cm.macro_f1())

    def test_f1_differ(self):
        self.assertNotAlmostEquals(self.cm.macro_f1(), self.cm.f1_from_avg_precision_recall(), 8)

    def test_task4_a(self):
        cm_model_a = ConfusionMatrix(["classA", "classB"])
        cm_model_a.increase_value("classA", "classA", 990)
        cm_model_a.increase_value("classB", "classA", 10)
        print(cm_model_a.get_matrix_copy())
        print(cm_model_a.accuracy())

        self.assertAlmostEqual(cm_model_a.accuracy(), 0.99, 3)
        print(cm_model_a.macro_f1())

    def test_task4_b(self):
        cm_model_b = ConfusionMatrix(["classA", "classB"])
        cm_model_b.increase_value("classA", "classA", 10495)
        cm_model_b.increase_value("classA", "classB", 10495)
        cm_model_b.increase_value("classB", "classA", 5)
        cm_model_b.increase_value("classB", "classB", 5)
        print(cm_model_b.get_matrix_copy())
        print(cm_model_b.accuracy())
        # 0.34201868666929863
        # 0.33375614387120617

        self.assertAlmostEqual(cm_model_b.accuracy(), 0.5, 3)

        print(cm_model_b.macro_f1())


