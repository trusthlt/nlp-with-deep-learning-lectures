import unittest
from sbert_project.sbert_most_similar import find_most_similar


class TestSBERTMostSimilar(unittest.TestCase):

    def test_most_similar(self):
        query_sentence = "A fast, dark-colored fox leaps over a sleepy canine."
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "A man is playing a guitar.",
            "The sun is shining brightly.",
            "A dog is sleeping on the couch."
        ]
        most_similar_sentence, similarity_score = find_most_similar(query_sentence, sentences)
        self.assertEqual(most_similar_sentence, "The quick brown fox jumps over the lazy dog.")
        self.assertGreater(similarity_score, 0.7)


if __name__ == '__main__':
    unittest.main()
