from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def cluster_sentences(sentences, num_clusters):
    if not sentences or num_clusters <= 0:
        raise ValueError("Sentences list must not be empty and number of clusters must be positive.")

    clusters = {}

    return clusters
