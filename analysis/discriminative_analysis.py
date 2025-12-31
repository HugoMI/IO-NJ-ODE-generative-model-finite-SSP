"""
authors: Hugo Martinez Ibarra

Implementation of the discriminative analysis to detect real and generated paths.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np

import random
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

# This path can be modified to point to the desired analysis folder
# dependding where the present code is being run from
analysis_path = ''
input_data_path = '{}input_data/'.format(analysis_path)
output_data_path = '{}output_data/'.format(analysis_path)

import os

# =====================================================================================================================

def makedirs(dirname):
    """Create directory if it does not exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# =====================================================================================================================

# Preparation of data
def preprocess_data(real_data, gen_data):
    """
    It prepares the data, in a matrix, for the discriminative analysis.
    
    :param real_data: np.ndarray, shape (nb_paths, nb_states, timesteps), real paths
    :param gen_data: np.ndarray, shape (nb_paths, nb_states, timesteps), real paths
    :return: X (samples), y (labels) for discriminative analysis
    """
    real_paths = torch.from_numpy(real_data[:, 2, :])
    gen_paths = torch.from_numpy(gen_data[:, :, 1])
    # Use the line below only to test the real data against itself
    # gen_paths = torch.from_numpy(gen_data[:, 2, :])
    n_paths = real_paths.shape[0]
    X = torch.cat([real_paths, gen_paths], dim=0).cpu().numpy()
    y = np.concatenate([np.ones(n_paths), np.zeros(n_paths)])
    return X, y

# Logistic Regression, RF, Gradient Boosting
def sklearn_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, prob)
        mean_prob = prob.mean()
        results[name] = {"AUC": auc, "MeanProb": mean_prob}
    return results

# Neural Network (FFN)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(X, y, epochs=10, lr=1e-3, device="cpu"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test  = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test, dtype=torch.float32, device=device)

    model = MLP(input_dim=X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for _ in range(epochs):
        opt.zero_grad()
        out = model(X_train).squeeze()
        loss = loss_fn(out, y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        prob = model(X_test).squeeze().cpu().numpy()
    auc = roc_auc_score(y_test.cpu().numpy(), prob)
    mean_prob = prob.mean()
    return {"AUC": auc, "MeanProb": mean_prob}

# RNN (GRU)
class RNNClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        _, h = self.rnn(x)
        return self.sigmoid(self.fc(h.squeeze(0)))


def train_rnn(X, y, epochs=10, lr=1e-3, device="cpu", batch_size=64):
    # split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test  = torch.tensor(X_test, dtype=torch.float32, device=device).unsqueeze(-1)
    y_test  = torch.tensor(y_test, dtype=torch.float32, device=device)

    # dataloader for batching
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = RNNClassifier().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # training loop
    for _ in range(epochs):
        for xb, yb in train_dl:
            opt.zero_grad()
            out = model(xb).squeeze()
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        prob = model(X_test).squeeze().cpu().numpy()
    auc = roc_auc_score(y_test.cpu().numpy(), prob)
    mean_prob = prob.mean()
    return {"AUC": auc, "MeanProb": mean_prob}


# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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

    
    def discrriminative_analysis(real_data, fake_data):

        print("Data shapes:", "REAL:", real_data.shape, "FAKE:", fake_data.shape)

        X, y = preprocess_data(real_data, fake_data)

        all_results = {}

        print("=== Sklearn models ===")
        results = sklearn_models(X, y)
        for k,v in results.items():
            all_results[k] = v
            print(k, v)

        print("\n=== MLP ===")
        fnn_results = train_mlp(X, y, device=device)
        all_results["FNN"] = fnn_results
        print(train_mlp(X, y, device=device))

        print("\n=== RNN ===")
        rnn_results = train_mlp(X, y, device=device)
        all_results["RNN"] = rnn_results
        print(train_rnn(X, y, device=device))

        all_results_df = pd.DataFrame(all_results).T
        print(all_results_df)

        return all_results_df

    discr_analysis_bm = discrriminative_analysis(real_data_bm, fake_data_bm)
    discr_analysis_rp = discrriminative_analysis(real_data_rp, fake_data_rp)

    discr_analysis = {"BM" : discr_analysis_bm, "RP" : discr_analysis_rp}

    for df_name in discr_analysis:
        discr_analysis[df_name].to_csv('{}discriminative_analysis_{}.csv'.format(output_data_path, df_name)) 




