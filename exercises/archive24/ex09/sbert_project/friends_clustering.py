import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def cluster_sentences(sentences, num_clusters):
    if not sentences or num_clusters <= 0:
        raise ValueError("Sentences list must not be empty and number of clusters must be positive.")

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(sentence_embeddings)

    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[i])

    return clusters, sentence_embeddings, kmeans.labels_


def visualize_clusters(sentences, num_clusters):
    clusters, sentence_embeddings, labels = cluster_sentences(sentences, num_clusters)

    # Reduce dimensions to 2D for visualization using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(sentence_embeddings)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette="viridis")

    for i, sentence in enumerate(sentences):
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], sentence, fontsize=9)

    plt.title("Friends TV Show Sentence Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster")
    plt.show()


if __name__ == "__main__":
    from convokit import Corpus, download

    corpus = Corpus(filename=download("friends-corpus"))
    # todo convert to sentences


    num_clusters = 5  # You can adjust the number of clusters as needed
    visualize_clusters(sentences, num_clusters)
