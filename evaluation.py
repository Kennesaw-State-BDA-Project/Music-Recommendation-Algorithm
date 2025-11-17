import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score


# Load Data

# Change for data set path
df = pd.read_csv(r"G:\\BigDataAnalytics\\Music-Recommendation-Algorithm-main\\Music-Recommendation-Algorithm-main\\processed_user_item_matrix.csv")

user_ids = df['user_id']
ratings_matrix = df.drop(columns=['user_id'])

# Helper Functions
def precision_at_k(true, pred, k=10):
    # Precision@K: fraction of recommended items that are relevant.
    pred_top_k = np.argsort(pred)[::-1][:k]
    true_top_k = np.argsort(true)[::-1][:k]
    relevant = set(true_top_k)
    recommended = set(pred_top_k)
    return len(relevant & recommended) / len(recommended)

def recall_at_k(true, pred, k=10):
    # Recall@K: fraction of relevant items that are recommended.
    pred_top_k = np.argsort(pred)[::-1][:k]
    true_top_k = np.argsort(true)[::-1][:k]
    relevant = set(true_top_k)
    recommended = set(pred_top_k)
    return len(relevant & recommended) / len(relevant)

def f1_at_k(true, pred, k=10):
    p = precision_at_k(true, pred, k)
    r = recall_at_k(true, pred, k)
    return 2 * (p * r) / (p + r + 1e-9)


# Baseline 1: Random Recommendation
def random_recommendation(user_vector, k=10):
    return np.random.rand(len(user_vector))

# Baseline 2: K-Means Clustering
def kmeans_recommendation(X, user_index, k=3):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    cluster = labels[user_index]
    cluster_users = X[labels == cluster]
    return cluster_users.mean(axis=0)

# Model: Collaborative Filtering (Simple User-User)
def collaborative_filtering(X, user_index, k=10):
    # Cosine similarity between users
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(X)
    user_sims = sims[user_index]
    # Weighted sum of ratings from similar users
    scores = user_sims @ X / (user_sims.sum() + 1e-9)
    return scores


# Evaluation Loop
X = ratings_matrix.values
results = {"Collaborative": [], "KMeans": [], "Random": []}

for user_index in range(min(50, len(X))):  # sample 50 users for speed
    true = X[user_index]

    # Collaborative Filtering
    pred_cf = collaborative_filtering(X, user_index)
    results["Collaborative"].append((precision_at_k(true, pred_cf),
                                     recall_at_k(true, pred_cf),
                                     f1_at_k(true, pred_cf)))
    # K-Means
    pred_km = kmeans_recommendation(X, user_index)
    results["KMeans"].append((precision_at_k(true, pred_km),
                              recall_at_k(true, pred_km),
                              f1_at_k(true, pred_km)))
    # Random
    pred_rand = random_recommendation(true)
    results["Random"].append((precision_at_k(true, pred_rand),
                              recall_at_k(true, pred_rand),
                              f1_at_k(true, pred_rand)))

# Results
def summarize(name, scores):
    precisions = [s[0] for s in scores]
    recalls = [s[1] for s in scores]
    f1s = [s[2] for s in scores]
    print(f"\n{name} Results:")
    print(f"Precision@K: {np.mean(precisions):.3f}")
    print(f"Recall@K:    {np.mean(recalls):.3f}")
    print(f"F1@K:        {np.mean(f1s):.3f}")

for model, scores in results.items():
    summarize(model, scores)
