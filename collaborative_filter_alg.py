import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the processed user–item matrix
def load_matrix(path="processed_user_item_matrix.csv"):
    df = pd.read_csv(path)
    df = df.set_index('user_id')
    return df

# Compute user-to-user similarity
def compute_similarity(matrix):
    similarity = cosine_similarity(matrix)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

# Predict scores for all user–song pairs
def predict_scores(matrix, similarity):
    sim_matrix = similarity.values
    rating_matrix = matrix.values

    # Weighted sum of neighbor ratings
    weighted_scores = sim_matrix.dot(rating_matrix)

    # Normalize by total similarity to avoid bias
    sim_sums = np.abs(sim_matrix).sum(axis=1).reshape(-1, 1)
    sim_sums[sim_sums == 0] = 1

    predictions = weighted_scores / sim_sums
    return pd.DataFrame(predictions, index=matrix.index, columns=matrix.columns)

# Recommend top-K songs the user has not listened to yet
def recommend_top_k(predictions, user_id, k=10):
    user_ratings = predictions.loc[user_id]

    # Only recommend songs with predicted > 0 and not already rated
    unseen_mask = (user_ratings == 0)
    recommendations = user_ratings[unseen_mask].sort_values(ascending=False).head(k)

    return recommendations
