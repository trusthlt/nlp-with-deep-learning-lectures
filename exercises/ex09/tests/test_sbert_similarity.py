import unittest
from sbert_project.sbert_similarity import compute_similarity


class TestSBERTSimilarity(unittest.TestCase):

    def test_similarity(self):
        sentence1 = "The quick brown fox jumps over the lazy dog."
        sentence2 = "A fast, dark-colored fox leaps over a sleepy canine."
        similarity = compute_similarity(sentence1, sentence2)
        self.assertGreater(similarity, 0.7)

    def test_dissimilarity(self):
        sentence1 = "The quick brown fox jumps over the lazy dog."
        sentence2 = "A man is playing a guitar."
        similarity = compute_similarity(sentence1, sentence2)
        self.assertLess(similarity, 0.5)


if __name__ == '__main__':
    unittest.main()
