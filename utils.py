# utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# -----------------------
# Data loading & preprocess
# -----------------------
def load_excel(path):
    """Load excel into DataFrame."""
    return pd.read_excel(path)

def preprocess(df):
    """
    Keep expected columns if present: length, protocal, info, source, destination, Time
    Returns X dataframe (numeric encoded), list of feature columns, original df.
    """
    df = df.copy()
    # Standardize column names: allow common minor typos
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc.startswith('proto') or lc == 'protocal':
            col_map[c] = 'protocal'
        elif lc == 'length':
            col_map[c] = 'length'
        elif lc == 'info':
            col_map[c] = 'info'
        elif lc == 'source':
            col_map[c] = 'source'
        elif lc == 'destination':
            col_map[c] = 'destination'
        elif lc.startswith('time'):
            col_map[c] = 'Time'
    df = df.rename(columns=col_map)
    # Columns to keep
    keep = [c for c in ['length','protocal','info','source','destination','Time'] if c in df.columns]
    data = df[keep].copy()
    # Convert Time to epoch numeric
    if 'Time' in data.columns:
        try:
            data['Time'] = pd.to_datetime(data['Time'])
            data['Time_epoch'] = (data['Time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        except Exception:
            data['Time_epoch'] = pd.to_numeric(data['Time'], errors='coerce').fillna(0)
    # Encode categorical columns (simple factorize)
    enc_cols = []
    for c in ['protocal','source','destination','info']:
        if c in data.columns:
            data[c] = data[c].astype(str).fillna("NA")
            data[c + '_enc'] = pd.factorize(data[c])[0]
            enc_cols.append(c + '_enc')
    # Feature columns: encodings + length + time_epoch if present
    feat_cols = [c for c in ['length','Time_epoch'] if c in data.columns] + enc_cols
    X = data[feat_cols].fillna(0).reset_index(drop=True)
    return X, feat_cols, df.reset_index(drop=True)

# -----------------------
# Label creation (simulate attacks if not present)
# -----------------------
def create_labels(df, X, label_col='attack'):
    """
    If df contains label_col, use it.
    Else simulate binary attack label based on rules.
    Returns y (pd.Series).
    """
    if label_col in df.columns:
        y = df[label_col].astype(int).reset_index(drop=True)
        return y
    # Rule-based simulation:
    score = np.zeros(len(X), dtype=float)
    if 'length' in X.columns:
        score += (X['length'] > X['length'].quantile(0.90)).astype(float) * 1.5
    enc_cols = [c for c in X.columns if c.endswith('_enc')]
    for c in enc_cols:
        value_counts = X[c].value_counts(normalize=True)
        rare_vals = value_counts[value_counts < 0.05].index.tolist()
        score += X[c].isin(rare_vals).astype(float) * 1.0
    score += np.random.rand(len(X)) * 0.5
    y = (score > np.percentile(score, 85)).astype(int)
    return pd.Series(y)

# -----------------------
# Model training (single client)
# -----------------------
def train_local_lgb(X, y, num_rounds=100, seed=42):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': seed
    }
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=num_rounds)
    return model

# -----------------------
# Federated simulation: train N client models by splitting X_train
# -----------------------
def train_federated_ensemble(X, y, n_clients=3, test_size=0.2, random_state=42):
    """
    Splits data into train/test, partitions train into n_clients, trains a LightGBM per client.
    Returns dict with models list, X_train/test, y_train/test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    idx_splits = np.array_split(X_train.index.to_numpy(), n_clients)
    models = []
    explainers = []
    for i, idxs in enumerate(idx_splits):
        Xi = X_train.loc[idxs]
        yi = y_train.loc[idxs]
        # if client has tiny data, we still train but maybe fewer rounds
        m = train_local_lgb(Xi, yi, num_rounds=100, seed=seed_from(i, random_state))
        models.append(m)
        explainers.append(shap.TreeExplainer(m))
    return {
        'models': models,
        'explainers': explainers,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def seed_from(index, base):
    return int(base) + int(index) * 7

# -----------------------
# Ensemble predict (average probabilities)
# -----------------------
def ensemble_predict_proba(models, X_input):
    preds = [m.predict(X_input) for m in models]
    preds = np.vstack(preds)
    return preds.mean(axis=0)

# -----------------------
# SHAP utilities: compute per-client shap and average
# -----------------------
def compute_avg_shap_for_samples(explainers, models, X_samples):
    """
    For a list of explainers/models compute SHAP values on X_samples and return mean across clients.
    Handles tree explainers which may return list-of-arrays for binary classification.
    """
    shap_vals_clients = []
    for expl, m in zip(explainers, models):
        sv = expl.shap_values(X_samples)
        # If sv is list (binary), take class 1 array if present
        if isinstance(sv, list) or (isinstance(sv, np.ndarray) and sv.ndim == 3):
            # shap may return [class0, class1] for binary - choose class1
            try:
                arr = sv[1]
            except Exception:
                # fallback: take last index
                arr = sv[-1]
        else:
            arr = sv
        arr = np.array(arr)
        # ensure shape (n_samples, n_features)
        shap_vals_clients.append(arr)
    # average
    avg = np.mean(np.stack(shap_vals_clients), axis=0)
    return avg

# -----------------------
# Plot helpers (save to file)
# -----------------------
def plot_shap_beeswarm(avg_shap, X_samples, out_path):
    plt.figure(figsize=(8,6))
    shap.summary_plot(avg_shap, X_samples, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_shap_bar(avg_shap, X_samples, out_path):
    plt.figure(figsize=(8,4))
    shap.summary_plot(avg_shap, X_samples, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_force_plot_html(explainer, shap_vals_row, row_X, out_html_path):
    """
    Save a force plot for one row. explainer may be any TreeExplainer.
    shap_vals_row: 1D array of shap values for that row (class 1)
    row_X: single-row DataFrame or Series
    """
    # ensure expected_value: pick explainer.expected_value for class 1 if needed
    ev = getattr(explainer, 'expected_value', None)
    if isinstance(ev, (list, np.ndarray)):
        base = ev[1] if len(ev) > 1 else ev[-1]
    else:
        base = ev
    fp = shap.force_plot(base, shap_vals_row, row_X, matplotlib=False)
    shap.save_html(out_html_path, fp)

# -----------------------
# Attack simulation helper
# -----------------------
def simulate_attack_on_row(orig_row, feat_cols, method='increase_length_and_make_rare'):
    """
    Given a single-row Series or DataFrame, create a synthetic attacked variant.
    Returns new_row (Series) and description.
    Methods:
      - increase_length_and_make_rare: multiply length by factor and flip one categorical enc to a rare id (-1)
    """
    new = orig_row.copy()
    desc = []
    if 'length' in feat_cols:
        # bump length by 2x to simulate large payload
        new['length'] = float(new['length']) * 2.5
        desc.append("length increased 2.5x")
    # For encoded categorical columns, set to a high (rare) index by adding large offset
    cat_cols = [c for c in feat_cols if c.endswith('_enc')]
    if cat_cols:
        # pick one category and set to a rare value (max+100)
        c = cat_cols[0]
        new[c] = int(new[c]) + 100
        desc.append(f"{c} set to rare value")
    return new, "; ".join(desc)

# -----------------------
# Save/load models
# -----------------------
def save_models(models, folder='models'):
    os.makedirs(folder, exist_ok=True)
    for i, m in enumerate(models):
        joblib.dump(m, os.path.join(folder, f'client_model_{i}.pkl'))

def load_models(folder='models'):
    models = []
    for fname in os.listdir(folder):
        if fname.endswith('.pkl'):
            models.append(joblib.load(os.path.join(folder, fname)))
    return models
