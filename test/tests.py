import sys
sys.path.append('/Users/antonskvarskij/Collocations')
import numpy as np
from utility.bert_vectors import BERTVectorizer
from utility.similarities import VectorSimilarities

vectorizer = BERTVectorizer()

def test_bert_vectors(text: str) -> np.ndarray:
    return vectorizer.get_bert_vectors(text)

def sim_check(vec1: np.ndarray, vec2: np.ndarray):
    similarities = VectorSimilarities()
    return (
    print("Cosine Similarity:", similarities.cosine_similarity(vec1, vec2)), 
    print("Euclidean Distance:", similarities.euclidean_distance(vec1, vec2)),
    print("Manhattan Distance:", similarities.manhattan_distance(vec1, vec2)),
    print("Jaccard Similarity:", similarities.jaccard_similarity(vec1, vec2)),
    print("Pearson Correlation:", similarities.pearson_correlation(vec1, vec2))
    )

if __name__ == "__main__":
    text = "ремонт это шляпа!"
    print(vectorizer)
    print(test_bert_vectors(text))
    
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])

    sim_check(vec1, vec2)