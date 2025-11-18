import runpy
import os
import sys

# ----------------------------
# Configuration / Paths
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_FILE = os.path.join(PROJECT_ROOT, "user_music_data.csv")
PROCESSED_DATA_FILE = os.path.join(PROJECT_ROOT, "processed_user_item_matrix.csv")

# Optional: number of users to evaluate
EVAL_SAMPLE_USERS = 50

# ----------------------------
# Step 1: Preprocessing
# ----------------------------
print("\n>>> STEP 1: Preprocessing raw data")
try:
    mod = runpy.run_path(os.path.join(PROJECT_ROOT, "music_data_preprocessing.py"))
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"Preprocessing failed or produced no output at {PROCESSED_DATA_FILE}")
    else:
        print(f"Preprocessing completed. Processed file: {PROCESSED_DATA_FILE}")
except Exception as e:
    print(f"Preprocessing failed: {e}")

# ----------------------------
# Step 2: K-Means
# ----------------------------
print("\n>>> STEP 2: Running K-Means script (kmeans.py)")
try:
    runpy.run_path(os.path.join(PROJECT_ROOT, "kmeans.py"))
    print("kmeans.py completed successfully.")
except Exception as e:
    print(f"kmeans.py execution failed: {e}")

# ----------------------------
# Step 3: Evaluation
# ----------------------------
print("\n>>> STEP 3: Running evaluation script (evaluation.py)")
try:
    # Ensure evaluation script reads the processed CSV locally
    eval_globals = {"PROCESSED_FILE": PROCESSED_DATA_FILE, "SAMPLE_USERS": EVAL_SAMPLE_USERS}
    runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation.py"), init_globals=eval_globals)
except Exception as e:
    print(f"evaluation.py execution failed: {e}")

# ----------------------------
# Step 4: Collaborative Filtering
# ----------------------------
print("\n>>> STEP 4: Running Collaborative Filtering (collaborative_filter_alg.py)")
try:
    runpy.run_path(os.path.join(PROJECT_ROOT, "collaborative_filter_alg.py"))
except Exception as e:
    print(f"Collaborative filtering failed: {e}")

# ----------------------------
# Step 5: Randomization (Optional)
# ----------------------------
print("\n>>> STEP 5: Running Randomization (random_algorithm.py)")
try:
    runpy.run_path(os.path.join(PROJECT_ROOT, "random_algorithm.py"))
except Exception as e:
    print(f"Randomization failed: {e}")

print("\n=== Pipeline complete ===")
