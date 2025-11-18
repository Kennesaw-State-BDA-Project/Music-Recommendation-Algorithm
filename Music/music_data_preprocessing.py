import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load Data
def load_data(path: str):
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    return df


# Preprocess Data
def preprocess_data(df: pd.DataFrame):
    df.columns = [c.lower() for c in df.columns]

    if {'user_id', 'song_id', 'rating'}.issubset(df.columns):
        # Flat format
        print("Detected flat interaction format.")

        df = df.dropna(subset=['user_id', 'song_id'])
        df['rating'] = df['rating'].fillna(0)

        df['user_id'] = df['user_id'].astype(str)
        df['song_id'] = df['song_id'].astype(str)
        df['rating'] = df['rating'].astype(float)

        if df['rating'].max() > 5:
            scaler = MinMaxScaler(feature_range=(0, 5))
            df['rating'] = scaler.fit_transform(df[['rating']])

        user_item_matrix = df.pivot_table(
            index='user_id',
            columns='song_id',
            values='rating',
            fill_value=0
        )
    else:
        # Pivoted format
        print("Detected pre-pivoted user-item matrix format.")

        df = df.rename(columns={df.columns[0]: 'user_id'})
        df['user_id'] = df['user_id'].astype(str)
        df = df.set_index('user_id')

        if df.max().max() > 5:
            scaler = MinMaxScaler(feature_range=(0, 5))
            df[df.columns] = scaler.fit_transform(df)

        user_item_matrix = df

    print("Preprocessing complete.")
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    return user_item_matrix

# Run preprocessing
if __name__ == "__main__":
    data_path = "user_item_matrix.csv"
    df = load_data(data_path)
    user_item_matrix = preprocess_data(df)

    # Save processed matrix
    user_item_matrix.to_csv("processed_user_item_matrix.csv")
    print("\nFile saved as 'processed_user_item_matrix.csv'.")