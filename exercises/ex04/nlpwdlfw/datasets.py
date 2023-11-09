from typing import List
from random import Random


class BinaryClassificationExample:
    def __init__(self, feature_vector: List[float], label=float):
        assert label == 0.0 or label == 1.0
        self.feature_vector = feature_vector
        self.label = label


class BinaryClassificationDataset:

    def get_training_examples(self) -> List[BinaryClassificationExample]:
        raise NotImplementedError()

    def get_test_examples(self) -> List[BinaryClassificationExample]:
        raise NotImplementedError()


class SyntheticBinaryClassificationDataset(BinaryClassificationDataset):

    def __init__(self):
        self.random_gen = Random(1234)

    def __generate_2n_examples(self, n: int = 100) -> List[BinaryClassificationExample]:
        # Generate n positive and n negative examples
        # Positive examples are centered around (1,1), negative around (3,3)
        result = []
        sigma = 0.5
        for _ in range(n):
            result.append(BinaryClassificationExample(
                [self.random_gen.gauss(1.0, sigma), self.random_gen.gauss(1.0, sigma)],
                1))
        for _ in range(n):
            result.append(BinaryClassificationExample(
                [self.random_gen.gauss(3.0, sigma), self.random_gen.gauss(3.0, sigma)],
                0))
        self.random_gen.shuffle(result)
        return result

    def get_training_examples(self) -> List[BinaryClassificationExample]:
        return self.__generate_2n_examples(100)

    def get_test_examples(self) -> List[BinaryClassificationExample]:
        return self.__generate_2n_examples(10)
