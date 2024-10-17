from typing import List
import numpy as np


class ConfusionMatrix:
    def __init__(self, classes: List[str]):
        # zero-filled matrix
        self.matrix = np.zeros((len(classes), len(classes)))

        # map each gold-standard class to an index (for confusion matrix indexing)
        self.mapping_name_to_index = {}
        for class_name, index in enumerate(classes):
            self.mapping_name_to_index[index] = class_name

    def get_matrix_copy(self) -> np.ndarray:
        return self.matrix.copy()

    def increase_value(self, gold: str, pred: str, increment: int = 1):
        ind_gold = self.mapping_name_to_index[gold]
        ind_pred = self.mapping_name_to_index[pred]

        self.matrix[(ind_gold, ind_pred)] += increment

    def precision_all_classes(self) -> np.ndarray:
        sum_of_columns = self.matrix.sum(axis=0)
        return self.matrix.diagonal() / sum_of_columns

    def recall_all_classes(self) -> np.ndarray:
        sum_of_rows = self.matrix.sum(axis=1)
        return self.matrix.diagonal() / sum_of_rows

    def accuracy(self) -> float:
        return self.matrix.diagonal().sum() / self.matrix.sum()

    def precision(self, class_name: str) -> float:
        ind_pred = self.mapping_name_to_index[class_name]
        return self.precision_all_classes()[ind_pred]

    def recall(self, class_name: str) -> float:
        ind_pred = self.mapping_name_to_index[class_name]
        return self.recall_all_classes()[ind_pred]

    def f1_score_all_classes(self) -> np.ndarray:
        precisions = self.precision_all_classes()
        recalls = self.recall_all_classes()

        # we might have divided by zero: fix nans and replace by zero (by convention)
        numerator = 2 * precisions * recalls
        denominator = precisions + recalls

        result = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=result, where=denominator != 0)

        return result

    def macro_f1(self) -> float:
        return self.f1_score_all_classes().mean()

    def f1_from_avg_precision_recall(self) -> float:
        numerator = 2 * self.precision_all_classes().mean() * self.recall_all_classes().mean()
        denominator = self.precision_all_classes().mean() + self.recall_all_classes().mean()

        # avoid division by zero
        result = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=result, where=denominator != 0)

        return result

