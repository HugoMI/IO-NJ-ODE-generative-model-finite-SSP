"""
authors: Hugo Martinez Ibarra

Implementation of the predictive analysis to assess the temporal strcuture of real and generated paths.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import random
import torch

import pandas as pd

# This path can be modified to point to the desired analysis folder
# dependding where the present code is being run from
analysis_path = ''
input_data_path = analysis_path + 'input_data/'
output_data_path = analysis_path + 'output_data/'

import os

# =====================================================================================================================

def makedirs(dirname):
    """Create directory if it does not exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# =====================================================================================================================

# Data preparation for the Time-Conditioned GLM
def make_features_time_conditional(seq, p=10):
    """
    Construct lag, run-length, and time features for a binary sequence.

    Parameters
    ----------
    seq : array-like of shape (T,)
        Binary sequence (0/1).
    p : int, default=10
        Number of lagged states to include as features.

    Returns
    -------
    X : ndarray, shape (T - p - 1, p + 2)
        Feature matrix (lags + run-length + time).
    y : ndarray, shape (T - p - 1,)
        Next-state binary targets.
    """
    x = np.asarray(seq, dtype=int)
    T = len(x)

    # Absolute time index
    time = np.arange(T)

    # Compute run-length (time since last switch)
    switches = np.r_[True, x[1:] != x[:-1]]
    runlen = np.zeros_like(x, dtype=int)
    cur = 0
    for t in range(T):
        if switches[t]:
            cur = 0
        runlen[t] = cur
        cur += 1

    # Lag features
    lags = np.column_stack([np.roll(x, i) for i in range(1, p + 1)])
    X = np.c_[lags, runlen, time]
    y = x

    # Drop first p rows (invalid lags) and last row (no y_next)
    valid = np.arange(p, T - 1)
    X = X[valid]
    y = y[valid + 1]
    return X, y


def combine_sequences_matrix(data, p=10):
    """
    Combine 2D array of sequences into a single feature/target matrix.

    Parameters
    ----------
    data : ndarray, shape (n_paths, length)
        Binary sequences (0/1).
    p : int, default=10
        Number of lags.

    Returns
    -------
    X : ndarray, shape (total_samples, p + 2)
    y : ndarray, shape (total_samples,)
    """
    Xs, ys = [], []
    for seq in data:
        X, y = make_features_time_conditional(seq, p=p)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


# Logistic Regression Training
def train_logistic(X, y):
    """
    Fit a time-conditioned logistic regression classifier with calibration.
    """
    base = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf = CalibratedClassifierCV(base, method='isotonic', cv=3)
    clf.fit(X, y)
    return clf


# Evaluation metrics
def evaluate_classifier(clf, X, y):
    """
    Evaluate classifier using log loss, Brier score, and accuracy.
    """
    prob = clf.predict_proba(X)[:, 1]
    y_pred = (prob >= 0.5).astype(int)
    return {
        "logloss": log_loss(y, prob),
        "brier": brier_score_loss(y, prob),
        "acc": accuracy_score(y, y_pred),
    }


# Predictive score (TSTR/TRTR)
def predictive_score_glm_time_conditional(reals, fakes, p=10, k=5):
    """
    Compute predictive score (|TSTR - TRTR|) using a time-conditioned GLM.

    Parameters
    ----------
    reals : ndarray, shape (n_real_paths, length)
        Real binary sequences (0/1).
    fakes : ndarray, shape (n_fake_paths, length)
        Fake binary sequences (0/1).
    p : int, default=10
        Number of lag features.
    k : int, default=5
        Number of folds for TRTR cross-validation.

    Returns
    -------
    score : float
        |TSTR_logloss - TRTR_logloss| predictive score (smaller = better).
    metrics : dict
        Dictionary with TRTR mean metrics and TSTR metrics.
    """

    # Build feature/target matrices
    X_real, y_real = combine_sequences_matrix(reals, p=p)
    X_fake, y_fake = combine_sequences_matrix(fakes, p=p)

    # --- TRTR (train real → test real) ---
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    trtr_metrics = []
    for train_idx, test_idx in kf.split(X_real):
        clf = train_logistic(X_real[train_idx], y_real[train_idx])
        m = evaluate_classifier(clf, X_real[test_idx], y_real[test_idx])
        trtr_metrics.append(m)
    trtr_mean = {key: np.mean([m[key] for m in trtr_metrics]) for key in trtr_metrics[0]}

    # --- TSTR (train fake → test real) ---
    clf_fake = train_logistic(X_fake, y_fake)
    tstr_metrics = evaluate_classifier(clf_fake, X_real, y_real)

    # Predictive score (smaller = better)
    score = abs(tstr_metrics["logloss"] - trtr_mean["logloss"])

    metrics = {"TRTR_mean": trtr_mean, "TSTR": tstr_metrics}
    return score, metrics


if __name__ == "__main__":
    def set_all_seeds(seed):
        """Set seeds for reproducibility."""
        # Python's built-in random module
        # random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch (CPU and CUDA)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # For multi-GPU
        
        # # Optional: For hash seed (affects reproducibility in some data shuffling/hashing)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        
        # # Optional: Configure PyTorch for deterministic operations (may impact performance)
        # # This is often crucial for full reproducibility in PyTorch, especially on GPUs
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Use the function to set the seed for your experiment
    SEED_VALUE = 42
    set_all_seeds(SEED_VALUE)

    print("Make sure the desired data is in the following folder:", input_data_path)
    makedirs(input_data_path)
    makedirs(output_data_path)

    fake_data_bm_filename = "data_future_1point_synth_bm.npy"
    # Use the line below only to test the real data against itself
    # fake_data_bm_filename = "data_npaths2_real_bm.npy"
    real_data_bm_filename = "data_npaths_real_bm.npy"

    fake_data_bm = np.load(input_data_path + f"{fake_data_bm_filename}")
    real_data_bm = np.load(input_data_path + f"{real_data_bm_filename}")

    fake_data_rp_filename = "data_future_1point_synth_rp.npy"
    # Use the line below only to test the real data against itself
    # fake_data_rp_filename = "data_npaths2_real_rp.npy"
    real_data_rp_filename = "data_npaths_real_rp.npy"

    fake_data_rp = np.load(input_data_path + f"{fake_data_rp_filename}")
    real_data_rp = np.load(input_data_path + f"{real_data_rp_filename}")

    def predictive_analysis(real_data, fake_data):

        print("Data shapes:", "REAL:", real_data.shape, "FAKE:", fake_data.shape)

        reals = real_data[:, 2, :]
        fakes = fake_data[:, :, 1]

        # Simulate nonstationary binary processes
        # reals = simulate_sign_bm(n_paths=50, length=300)
        # fakes = simulate_alternating_renewal(n_paths=50, length=300)

        # Compute predictive score using time-conditioned GLM
        score, metrics = predictive_score_glm_time_conditional(reals, fakes, p=10, k=5)

        print("\n=== Time-Conditioned GLM Predictive Score ===")
        print(f"Predictive Score (|TSTR - TRTR|) [logloss]: {score:.4f}\n")

        all_results_trtr = {}
        all_results_tstr = {}
        all_results  = {}

        print("TRTR mean metrics:")
        for k, v in metrics["TRTR_mean"].items():
            all_results_trtr[k] = v
            print(f"  {k}: {v:.4f}")
        all_results["TRTR"] = all_results_trtr
        

        print("\nTSTR metrics:")
        for k, v in metrics["TSTR"].items():
            all_results_tstr[k] = v
            print(f"  {k}: {v:.4f}")

        all_results["TSTR"] = all_results_tstr
        all_results["diff"] = {"pred_score" : score}

        all_results_df = pd.DataFrame(all_results)
        print(all_results_df)


        # # Optional: visualize coefficient magnitudes (use plain LogisticRegression)
        # Xr, yr = combine_sequences_matrix(reals, p=10)
        # plain_lr = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xr, yr)
        # coef = plain_lr.coef_[0]  # <- this works

        # feature_names = [f"lag{i}" for i in range(1, 11)] + ["runlen", "time"]

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 4))
        # plt.bar(feature_names, coef)
        # plt.xticks(rotation=45)
        # plt.title("Time-Conditioned GLM Coefficients (Real Data, Uncalibrated LR)")
        # plt.tight_layout()
        # plt.show()

        return all_results_df
    
    pred_analysis_bm = predictive_analysis(real_data_bm, fake_data_bm)
    pred_analysis_rp = predictive_analysis(real_data_rp, fake_data_rp)

    pred_analysis = {"BM" : pred_analysis_bm, "RP" : pred_analysis_rp}

    for df_name in pred_analysis:
        pred_analysis[df_name].to_csv('{}predictive_analysis_{}.csv'.format(output_data_path, df_name)) 

