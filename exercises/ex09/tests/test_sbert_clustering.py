import unittest
from sbert_project.sbert_clustering import cluster_sentences


class TestSBERTClustering(unittest.TestCase):

    def test_clustering(self):
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast, dark-colored fox leaps over a sleepy canine.",
            "A man is playing a guitar.",
            "The sun is shining brightly.",
            "A dog is sleeping on the couch."
        ]
        num_clusters = 2
        clusters = cluster_sentences(sentences, num_clusters)
        self.assertEqual(len(clusters), num_clusters)

    def test_empty_sentences(self):
        sentences = []
        num_clusters = 2
        with self.assertRaises(ValueError):
            cluster_sentences(sentences, num_clusters)

    def test_invalid_num_clusters(self):
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast, dark-colored fox leaps over a sleepy canine."
        ]
        num_clusters = 0
        with self.assertRaises(ValueError):
            cluster_sentences(sentences, num_clusters)


if __name__ == '__main__':
    unittest.main()
