import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Configuration (dynamic)
# ----------------------------
# Use PROCESSED_FILE and SAMPLE_USERS if provided by Driver; else default values
processed_file = globals().get("PROCESSED_FILE", "processed_user_item_matrix.csv")
sample_users = globals().get("SAMPLE_USERS", 50)
k_recommend = globals().get("K_RECOMMEND", 10)  # top-K items

if not os.path.exists(processed_file):
    raise FileNotFoundError(f"Processed CSV not found at {processed_file}")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(processed_file)
user_ids = df['user_id']
ratings_matrix = df.drop(columns=['user_id'])
X = ratings_matrix.values

print(f"Loaded user-item matrix: {X.shape[0]} users, {X.shape[1]} items")

# ----------------------------
# Evaluation metrics
# ----------------------------
def precision_at_k(true, pred, k=k_recommend):
    pred_top_k = np.argsort(pred)[::-1][:k]
    true_top_k = np.argsort(true)[::-1][:k]
    return len(set(pred_top_k) & set(true_top_k)) / len(pred_top_k)

def recall_at_k(true, pred, k=k_recommend):
    pred_top_k = np.argsort(pred)[::-1][:k]
    true_top_k = np.argsort(true)[::-1][:k]
    return len(set(pred_top_k) & set(true_top_k)) / len(true_top_k)

def f1_at_k(true, pred, k=k_recommend):
    p = precision_at_k(true, pred, k)
    r = recall_at_k(true, pred, k)
    return 2 * (p * r) / (p + r + 1e-9)

# ----------------------------
# Recommendation models
# ----------------------------
def random_recommendation(user_vector):
    return np.random.rand(len(user_vector))

def kmeans_recommendation(X, user_index, k_clusters=3):
    kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    cluster = labels[user_index]
    cluster_users = X[labels == cluster]
    return cluster_users.mean(axis=0)

def collaborative_filtering(X, user_index):
    sims = cosine_similarity(X)
    user_sims = sims[user_index]
    scores = user_sims @ X / (user_sims.sum() + 1e-9)
    return scores

# ----------------------------
# Run evaluation
# ----------------------------
results = {"Collaborative": [], "KMeans": [], "Random": []}

for user_index in range(min(sample_users, len(X))):
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

# ----------------------------
# Summarize results
# ----------------------------
def summarize(name, scores):
    precisions = [s[0] for s in scores]
    recalls = [s[1] for s in scores]
    f1s = [s[2] for s in scores]
    print(f"\n{name} Results:")
    print(f"Precision@{k_recommend}: {np.mean(precisions):.3f}")
    print(f"Recall@{k_recommend}:    {np.mean(recalls):.3f}")
    print(f"F1@{k_recommend}:        {np.mean(f1s):.3f}")

for model, scores in results.items():
    summarize(model, scores)
