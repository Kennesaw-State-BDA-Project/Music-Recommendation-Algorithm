import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs

df = pd.read_csv("processed_user_item_matrix.csv")
print(df.head())
