# model.py
from utils import train_local_lgb

# This file can stay tiny — or host alternative models (XGBoost, RF).
# Example: wrapper to train local model (kept for modularity)
def train_model(X, y, num_rounds=100, seed=42):
    return train_local_lgb(X, y, num_rounds=num_rounds, seed=seed)
