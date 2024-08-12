import numpy as np   


class VectorSimilarities:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def cosine_similarity(self, vec1, vec2):
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        return (np.dot(vec1, vec2) + self.eps) / (vec1_norm * vec2_norm  + self.eps)

    def euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2) + self.eps

    def manhattan_distance(self, vec1, vec2):
        return np.sum(np.abs(vec1 - vec2))  + self.eps

    def jaccard_similarity(self, vec1, vec2):
        intersection = np.minimum(vec1, vec2).sum() + self.eps
        union = np.maximum(vec1, vec2).sum() + self.eps
        return (intersection  + self.eps) / (union  + self.eps)

    def pearson_correlation(self, vec1, vec2):
        vec1_mean = vec1 - np.mean(vec1)
        vec2_mean = vec2 - np.mean(vec2)
        numerator = np.sum(vec1_mean * vec2_mean) + self.eps
        denominator = np.sqrt(np.sum(vec1_mean ** 2) * np.sum(vec2_mean ** 2) + self.eps)
        return numerator / denominator
