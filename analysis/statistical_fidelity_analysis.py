"""
authors: Hugo Martinez Ibarra

Implementation of the statistical fidelity metrics to compare real vs synthetic paths
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
import warnings
import os

data_path = '../data/'
training_data_path = '{}training_data/'.format(data_path)

# This path can be modified to point to the desired analysis folder
# dependding where the present code is being run from
analysis_path = ''
input_data_path = analysis_path + 'input_data/'
output_data_path = analysis_path + 'output_data/'

# =====================================================================================================================

def makedirs(dirname):
    """Create directory if it does not exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# =====================================================================================================================

def plot_real_paths(path_s, path_t, filename, samples=[0, 18, 41, 3, 20, 26]):
    """
    It plots 6 samples of real paths, im a 3x2 grid.
    
    :param path_s: np.ndarray, shape (n_samples, n_timesteps), paths to plot
    :param path_t: np.ndarray, shape (n_timesteps,), time points corresponding to the paths
    :param samples: list of int, indices of the samples to plot
    """

    datasets_plot = []

    n_j = 2
    n_i = 3

    for b in samples:
        path_s = real_paths[b, 2, :]
        datasets_plot.append([path_t, path_s])

    counter = 0
    fig, axlist = plt.subplots(n_i, n_j, figsize=(14, 8))
    for i in range(n_i):
        for j in range(n_j):
            axlist[i, j].step(datasets_plot[counter][0], datasets_plot[counter][1], color='#1f77b4', label=f'real {counter + 1}')

            axlist[i, j].legend()
            axlist[i, j].set_xlabel("$t$")
            axlist[i, j].set_ylabel("$V_{t, 2}=1_{\{S_{t}=1\}}$")
            counter += 1

    fig.tight_layout()
    fig.savefig("{}real_paths_samples_{}.png".format(output_data_path, filename), dpi=300)
    # plt.show()


makedirs(input_data_path)
makedirs(output_data_path)

model_name = "BMClassification2"
dataset_id = 1

model_name = "RPClassification"
dataset_id = 3

filename = f"{model_name}-{dataset_id}"


delta_t = 0.01

real_paths = np.load(training_data_path + f"{model_name}-{dataset_id}/data.npy")
real_path_t = np.arange(real_paths.shape[2])*delta_t

print("real_paths:", real_paths.shape, real_paths)
plot_real_paths(real_paths, real_path_t, filename)



def plot_predicted_paths(incomplete_past_path, complete_past_path, future_path, path_t, filename, samples=[0, 18, 41, 3, 20, 26]):
    """
    It plots 6 samples of real paths, im a 3x2 grid.
    
    :param incomplete_past_path: np.ndarray, shape (n_samples, n_timesteps, nb_states), path of observed (incomplete) data to plot
    :param complete_past_path: np.ndarray, shape (n_samples, n_timesteps, nb_states), path of complete data to plot
    :param future_path: np.ndarray, shape (n_samples, n_timesteps, nb_states), paths of complete data to plot
    :param path_t: np.ndarray, shape (n_timesteps,), time points corresponding to the future paths
    :param samples: list of int, indices of the samples to plot
    """

    datasets_plot = []

    n_j = 2
    n_i = 3

    for b in samples:
        path_s = future_path[b, :, 1]
        datasets_plot.append([path_t, path_s])

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    counter = 0
    fig, axlist = plt.subplots(n_i, n_j, figsize=(14, 8))
    for i in range(n_i):
        for j in range(n_j):
            axlist[i, j].step(complete_past_path[0,:], complete_past_path[3,:], color="#000000")
            axlist[i, j].scatter(incomplete_past_path[0,:], incomplete_past_path[3,:], color="#000000", label='observed', s=20)
            axlist[i, j].step(datasets_plot[counter][0], datasets_plot[counter][1], color=colors[counter], label=f'synthetic {counter + 1}')

            axlist[i, j].legend()
            axlist[i, j].set_xlabel("$t$")
            axlist[i, j].set_ylabel("$V_{t, 2}=1_{\{S_{t}=1\}}$")
            counter += 1

    fig.tight_layout()
    fig.savefig("{}synthetic_paths_samples_{}.png".format(output_data_path, filename), dpi=300)
    # plt.show()


print("Make sure the desired data is in the following folder:", input_data_path)
makedirs(input_data_path)
makedirs(output_data_path)

complete_past_paths_filename = "data_past_real_complete_bm.npy"
incomplete_past_paths_filename = "data_past_real_incomplete_bm.npy"
future_paths_filename = "data_future_seq_synth_bm.npy"

# complete_past_paths_filename = "data_past_real_complete_rp.npy"
# incomplete_past_paths_filename = "data_past_real_incomplete_rp.npy"
# future_paths_filename = "data_future_seq_synth_rp.npy"

future_times_filename = "future_times_past_seq.npy"

complete_past_paths = np.load(input_data_path + f"{complete_past_paths_filename}")
incomplete_past_paths = np.load(input_data_path + f"{incomplete_past_paths_filename}")
future_paths = np.load(input_data_path + f"{future_paths_filename}")
future_path_t = np.load(input_data_path + f"{future_times_filename}")

model_name = "BMClassification2"

plot_predicted_paths(
    incomplete_past_paths,
    complete_past_paths,
    future_paths,
    future_path_t,
    filename=model_name
)

def acf(X, lags=None):
    """
    Compute sample autocorrelation(s) for one or more time series, robust to zero variance.
    :param X: 1D or 2D array-like, shape (n_time,) or (n_series, n_time), time series data
    :param lags: int or array-like, lags at which to compute the ACF. If None, computes for all lags up to n_time-1.
    :return: array, shape (len(lags),) or (n_series, len(lags)), autocorrelation values
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[np.newaxis, :]

    n_series, n_time = X.shape

    # Define lags
    if lags is None:
        lags = np.arange(n_time)
    elif np.isscalar(lags):
        lags = np.array([lags])
    else:
        lags = np.asarray(lags)

    acf = np.full((n_series, len(lags)), np.nan)  # initialize with NaN

    means = np.mean(X, axis=1, keepdims=True)
    var_denom = np.sum((X - means) ** 2, axis=1)

    # Identify zero or NaN variance cases
    zero_var_mask = (var_denom == 0) | np.isnan(var_denom)
    if np.any(zero_var_mask):
        warnings.warn("Some time series have zero or NaN variance — their ACF will be set to NaN.")

    for k_idx, k in enumerate(lags):
        if k >= n_time:
            raise ValueError(f"Lag {k} exceeds time series length {n_time}")
        num = np.sum((X[:, :n_time - k] - means) * (X[:, k:] - means), axis=1)

        with np.errstate(invalid='ignore', divide='ignore'):
            acf[:, k_idx] = num / var_denom

        # Replace invalid entries (div by zero or NaN)
        acf[zero_var_mask, k_idx] = np.nan

    acf = np.nan_to_num(acf, nan=0.0)

    if acf.shape[0] == 1:
        return acf[0]
    return acf





#############################################################################################################################

def plot_metric_histograms(
        real_data,
        fake_data,
        filename,
        bins=30,
        lags_to_show=[25, 50, 75, 100],
    ):
    """
    It plots histograms for several metrics comparing real and fake data, including autocorrelation.
    
    :param real_data: np.ndarray, 
    :param fake_data: Description
    :param metric_names: Description
    :param bins: Description
    :param lags_to_show: Description
    """
    
    metric_names = ['mean', 'median', 'standard deviation', 'skewness', 'kurtosis']

    metric_funcs = {
        'mean': np.mean,
        'median': np.median,
        'standard deviation': np.std,
        'skewness': lambda x, axis: stats.skew(x, axis=axis),
        'kurtosis': lambda x, axis: stats.kurtosis(x, axis=axis),
    }

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # Plot main metrics in the first 5 cells
    for i, metric in enumerate(metric_names):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        func = metric_funcs[metric]
        real_metric = func(real_data, axis=1)
        fake_metric = func(fake_data, axis=1)
        wd = stats.wasserstein_distance(real_metric[~np.isnan(real_metric)], fake_metric[~np.isnan(fake_metric)])

        ax.hist(real_metric, bins=bins, alpha=0.6, label='Real', density=True)
        ax.hist(fake_metric, bins=bins, alpha=0.6, label='Synthetic', density=True)
        ax.set_title(f'{metric.capitalize()} (W-dist: {wd:.2e})', fontsize=13)
        ax.set_xlabel(f'{metric.capitalize()} value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        ax.text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"W-dist: {wd:.2e}", fontsize=10)

    # Last cell: 2x2 grid for autocorrelation lags, filling the bottom-right cell
    gs_acf = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2, 1], wspace=0.25, hspace=0.35)
    for lag_idx, lag in enumerate(lags_to_show):
        ax_acf = fig.add_subplot(gs_acf[lag_idx])
        real_acf = acf(real_data, lags=lag)
        fake_acf = acf(fake_data, lags=lag)
        wd = stats.wasserstein_distance(real_acf[~np.isnan(real_acf)], fake_acf[~np.isnan(fake_acf)])
        ax_acf.hist(real_acf, bins=bins, alpha=0.6, label='Real', density=True)
        ax_acf.hist(fake_acf, bins=bins, alpha=0.6, label='Synthetic', density=True)
        ax_acf.set_title(f'Autocorr lag {lag} (W-dist: {wd:.2e})', fontsize=11)
        ax_acf.tick_params(axis='both', which='major', labelsize=8)
        if lag_idx == 0:
            ax_acf.legend(fontsize=8)
        y_min, y_max = ax_acf.get_ylim()
        x_min, x_max = ax_acf.get_xlim()
        ax_acf.text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"W-dist: {wd:.2e}", fontsize=8)
    # plt.suptitle('Metric Distributions and Autocorrelation Lags', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("{}metric_histograms_{}.png".format(output_data_path, filename), dpi=300)
    # plt.show()


def plot_metric_evolution(real_data, fake_data, fake_times, filename):
    """
    Plots histograms for several metrics comparing real and fake data, including autocorrelation.
    Args:
        real_data: np.ndarray, shape (n_samples, n_timesteps)
        fake_data: np.ndarray, shape (n_samples, n_timesteps)
        metric_names: list of str, metrics to compute and plot
        bins: int, number of histogram bins
    """
    from scipy import stats

    metric_names = ['mean', 'median', 'standard deviation', 'skewness', 'kurtosis', 'autocorrelation']

    metric_funcs = {
        'mean': np.mean,
        'median': np.median,
        'standard deviation': np.std,
        'skewness': lambda x, axis: stats.skew(x, axis=axis),
        'kurtosis': lambda x, axis: stats.kurtosis(x, axis=axis),
        'autocorrelation': lambda x, axis: np.mean(acf(x), axis=axis)
    }

    n_metrics = len(metric_names)
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metric_names):
        print("Plotting metric:", metric)
        func = metric_funcs[metric]
        real_metric = func(real_data, axis=0)
        fake_metric = func(fake_data, axis=0)
        # print("real_metric", real_metric)
        # print("fake_metric", fake_metric)
        # wd = stats.wasserstein_distance(real_metric[~np.isnan(real_metric)], fake_metric[~np.isnan(fake_metric)])

        ax = axes[i]
        if metric != 'autocorrelation':
            ax.plot(fake_times, real_metric, label='Real')
            ax.plot(fake_times, fake_metric, label='Synthetic')
            ax.set_xlabel(f'$t$')
        else:
            ax.plot(np.arange(len(real_metric)), real_metric, label='Real')
            ax.plot(np.arange(len(fake_metric)), fake_metric, label='Synthetic')
            ax.set_xlabel(f'Time lag')
        ax.set_title(f'{metric.capitalize()} value through time', fontsize=13)
        ax.set_ylabel('Statistic value')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Optionally, add text inside the plot
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        # ax.text(0.05*(x_max-x_min)+x_min, 0.9*(y_max-y_min)+y_min, f"W-dist: {wd:.2e}", fontsize=10)

    # Hide unused subplots if any
    for j in range(i+1, n_rows*n_cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("{}metric_evolution_{}.png".format(output_data_path, filename), dpi=300)
    # plt.show()


print("Make sure the desired data is in the following folder:", input_data_path)
makedirs(input_data_path)
makedirs(output_data_path)

# For these experiments, always use 1-point observed pasts and synthetic futures generated accordingly
path_t_filename = "future_times_past_1point.npy"

fake_data_filename = "data_future_1point_synth_bm.npy"
real_data_filename = "data_npaths_real_bm.npy"

# fake_data_filename = "data_future_1point_synth_rp.npy"
# real_data_filename = "data_npaths_real_rp.npy"

path_t = np.load(input_data_path + f"{path_t_filename}")
fake_data = np.load(input_data_path + f"{fake_data_filename}")
real_data = np.load(input_data_path + f"{real_data_filename}")


print("Real data:", real_data.shape, real_data)
print("Fake data:", fake_data.shape, fake_data)

# This corresponds to the 2nd state variable (V_t,2 = 1_{S_t=1}) of the real data
real_paths = real_data[:, 2, :]
print("Real paths:", real_paths.shape, real_paths)

# This corresponds to the 2nd state variable (V_t,2 = 1_{S_t=1}) of the synthetic data
fake_paths = fake_data[:, :, 1]
print("Fake paths:", fake_paths.shape, fake_paths)

model_name = "BMClassification2"

plot_metric_histograms(real_paths, fake_paths, filename=model_name)
plot_metric_evolution(real_paths, fake_paths, path_t, filename=model_name)
