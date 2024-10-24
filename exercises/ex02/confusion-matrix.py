from statistics import mean
from typing import Dict, Set, List


class ConfusionMatrix:
    def __init__(self):
        self.labels: List[str] = []
        self.total_count: int = 0
        self.correct_count: int = 0
        self.matrix: Dict[str, Dict[str, int]] = dict()

    def increase(self, true_label: str, prediction: str, count: int = 1):
        if not true_label in self.matrix:
            self.matrix[true_label] = dict()
        if not prediction in self.matrix[true_label]:
            self.matrix[true_label][prediction] = 0

        ## add one
        self.matrix[true_label][prediction] += count
        print(self.matrix)

        if true_label == prediction:
            self.correct_count += count
        self.total_count += count

    def recall_for_label(self, label: str) -> float:
        # collect a row
        row: Dict[str, int] = self.matrix[label]
        denominator = float(sum(row.values()))
        nominator = float(row[label])
        return nominator / denominator

    def precision_for_label(self, label: str) -> float:
        # collect a column
        column_vals: List[int] = []
        for row in self.matrix.values():
            value: int = row[label]
            column_vals.append(value)

        denominator = float(sum(column_vals))
        nominator = float(self.matrix[label][label])
        if denominator == 0.0:
            return 0.0

        return nominator / denominator

    def f1_for_label(self, label: str)-> float:
        precision = self.precision_for_label(label)
        recall = self.recall_for_label(label)
        if precision == 0.0 and recall == 0.0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def accuracy(self) -> float:
        return float(self.correct_count) / float(self.total_count)

    def macro_f1(self) -> float:
        all_f1 = []
        for key in self.matrix.keys():
            all_f1.append(self.f1_for_label(key))
        
        return mean(all_f1)


if __name__ == "__main__":
    cf = ConfusionMatrix()
    cf.increase("NN", "NN", 25)
    cf.increase("NN", "VB", 5)
    cf.increase("NN", "ADJ", 1)
    cf.increase("VB", "NN", 2)
    cf.increase("VB", "VB", 15)
    cf.increase("VB", "ADJ", 12)
    cf.increase("ADJ", "NN", 1)
    cf.increase("ADJ", "VB", 6)
    cf.increase("ADJ", "ADJ", 0)


    print(cf.recall_for_label("NN"))
    print(cf.precision_for_label("NN"))
    print(cf.precision_for_label("ADJ"))
    print(cf.f1_for_label("ADJ"))
    print(cf.accuracy())

    print(cf.macro_f1())

    cf_model_one = ConfusionMatrix()
    cf_model_one.increase("classA", "classA", 990)
    cf_model_one.increase("classA", "classB", 0)
    cf_model_one.increase("classB", "classA", 10)
    cf_model_one.increase("classB", "classB", 0)
    print(cf_model_one.matrix)
    print(cf_model_one.accuracy())
    print(cf_model_one.macro_f1())

    cf_model_one = ConfusionMatrix()
    cf_model_one.increase("classA", "classA", 495)
    cf_model_one.increase("classA", "classB", 495)
    cf_model_one.increase("classB", "classA", 5)
    cf_model_one.increase("classB", "classB", 5)
    print(cf_model_one.matrix)
    print(cf_model_one.accuracy())
    print(cf_model_one.macro_f1())

