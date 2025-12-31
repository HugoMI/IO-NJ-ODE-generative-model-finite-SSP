"""
authors: Hugo Martinez Ibarra

implementation of the generative scheme for the IO NJ-ODE model
in the context of finite space processes
"""

import torch
import models
import json
import pandas as pd
import numpy as np
import os

from absl import app
from absl import flags

from torch.utils.data import DataLoader
import data_utils

from configs import config

from functools import reduce

# =====================================================================================================================

default_model_id = 1
default_prior_data_file_name = None
default_seed = 277
default_gpu = False
default_device = torch.device('cpu')

default_model_gen_dict = {
    'model_name': 'BMClassification2',
    'nb_gen_paths': 10,
    'nb_steps': 100,
    'delta_t': 0.01,
    'input_coords': [1, 2],
    'output_coords': [1, 2],
}

FLAGS = flags.FLAGS
flags.DEFINE_string("gen_dataset_params", None,
                    "name of the dict with hyper-params for the generative process")
flags.DEFINE_integer("model_id", default_model_id,
                    "id of the trined model to use")
flags.DEFINE_string("prior_data_file_name", None,
                    "name of the file (<name>.npy) which contains prior data")
flags.DEFINE_integer("gen_seed", default_seed,
                     "seed for making the synthetic dataset generation reproducible")
flags.DEFINE_bool("use_gpu", default_gpu, "whether to use GPU for generative process")

hyperparam_default = config.hyperparam_default
data_path = config.data_path

training_data_path = config.training_data_path
saved_models_path = f"{data_path}saved_models_"

input_data_gen_process_path = '{}input_data_generative_process/'.format(data_path)
output_data_gen_process_path = '{}output_data_generative_process/'.format(data_path)

print("++++++++++++++saved_models_path+++++++++++++", saved_models_path)
print("++++++++++++++input_data_gen_process_path+++++++++++++", input_data_gen_process_path)
print("++++++++++++++output_data_gen_process_path+++++++++++++", output_data_gen_process_path)

# =====================================================================================================================
def makedirs(dirname):
    """Create directory if it does not exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


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


def preprocessing_data(
    prior_data_file_name = default_prior_data_file_name,
    hyperparam_dict = default_model_gen_dict
):
    """
    Preprocessing of the prior data to create the
    starting data pack to initialize the generative process.
    :param prior_data_file_name: str, name of the file which contains prior data
    :param hyperparam_dict: dict, contains specifications for the desired synthetic data to be generated
    :return: dict, starting data pack to initialize the generative process
    """

    model_name = hyperparam_dict['model_name']
    nb_gen_paths = hyperparam_dict['nb_gen_paths']

    # Path of the prior data
    prior_data_file_path = '{}{}/{}'.format(input_data_gen_process_path, model_name, prior_data_file_name)
    path_data = np.load(prior_data_file_path)

    # print("path_data", path_data.shape, path_data)

    print("=====> Make sure that first dimension of prior data corresponds to time and starts at 0 <=====")
    # START ALWAYS FROM TIME 0 TO SEE IF THAT FIX SHIFT IN TIMES
    times  = path_data[0, :]
    T = times[-1]
    times_ext = times

    # print("Times from provided path:", times)
    # print("Last prior time point:", T)

    # These functions help to find the GCD of time differences
    # that will be set as the time step for the dense time grid
    # This is done since the time points provided can be irregularly spaced
    # and we need to create a dense time grid that includes all provided time points
    def numerical_gcd(a, b, tol=1e-9):
        """It finds the numerical GCD of two floats."""
        while abs(b) > tol:
            a, b = b, a % b
        return a

    def numerical_gcd_of_array(array, tol=1e-9):
        """It finds the numerical GCD of an array of floats."""
        if len(array) == 0:
            return 0
        # Use functools.reduce to apply the gcd function iteratively
        return reduce(lambda x, y: numerical_gcd(x, y, tol), array)

    # Time differences between consecutive time points
    # and appropiate time step for dense grid
    diffs = np.diff(times_ext)
    dt = numerical_gcd_of_array(diffs)

    print(f"The calculated time step (smallest commoon difference) is: {dt}")


    # Create the complete, dense time sequence using the GCD as the step.
    # Use round() to mitigate floating-point accumulation errors in np.arange().

    # This is the case when a sequence of points is provided as prior data
    if (len(times) > 1) and (dt != 0.):
        times_full = np.arange(times[0], times[-1], dt)

        # Use np.isclose() to handle floating point comparisons.
        # Reshape t_obs to broadcast correctly.
        obs_times_boolean = np.isclose(times_full[:, np.newaxis], times).any(axis=1)

        # Convert the boolean array to integers.
        obs_times_array = obs_times_boolean.astype(int)

        # print(f"The availability array is: {obs_times_array}")

        # print("SUM availability array", np.sum(obs_times_array))

        path_full = np.zeros((1, path_data.shape[0] - 1, len(times_full)))
        original_idx = 0
        for t in range(len(times_full)):
            if obs_times_array[t] == 1:
                path_full[:, :, t] = path_data[1:, original_idx]
                original_idx += 1

        observed_dates = np.repeat(np.reshape(obs_times_array, (1, -1)), nb_gen_paths, axis=0)

    # This is the case when a single one point is provided as prior data
    elif (len(times) == 1) and (dt == 0.):
        times_full = times
        path_full = path_data[1:, :]
        path_full = path_full[np.newaxis, :, :]
        # print("Case one point", path_full.shape, path_full)
        observed_dates = np.ones((nb_gen_paths, 1))
    else:
        raise ValueError("There might be repeated time stamp values '{}'.".format(times))
    
    print(f"Full time sequence: {times_full}")
    print("Full path", path_full.shape, path_full)

    idx = np.arange(nb_gen_paths)
    set_paths = np.repeat(path_full, nb_gen_paths, axis=0)
    nb_obs = np.sum(observed_dates, axis=1) - 1
    obs_noise = None


    dataloader_dict = {"idx": idx, "stock_path": set_paths[idx], 
                    "observed_dates": observed_dates[idx], 
                    "nb_obs": nb_obs[idx], "dt": dt,
                    "obs_noise": obs_noise}

    # This creates the dataset object, neeeded for the dataloader
    # which the model will use as input data
    input_data = data_utils.PriorDataset(dataset_dict=dataloader_dict)

    collate_fn, mult = data_utils.CustomCollateFnGen([])
    # N_DATASET_WORKERS = 0

    input_dl = DataLoader(
            dataset=input_data,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=nb_gen_paths,
            # num_workers=N_DATASET_WORKERS
    )

    # This contains the prior data plus some other parameters
    # needed to initialize the generative process,
    # including the neural ODE
    starting_data_pack = {"dataloader":input_dl,
                        "last_dt": dt,
                        "last_nb_paths": nb_gen_paths,
                        "last_times": times,
                        "last_T": T,
                        "model_name": model_name,
    }

    return starting_data_pack


def get_dataset_overview(data_path=output_data_gen_process_path):
    data_overview = '{}dataset_overview.csv'.format(
        data_path)
    makedirs(data_path)
    if not os.path.exists(data_overview):
        df_overview = pd.DataFrame(
            data=None, columns=['name', 'id', 'description'])
    else:
        df_overview = pd.read_csv(data_overview, index_col=0)
    return df_overview, data_overview


def generative_process(
    starting_data_pack,
    model_id = default_model_id,
    hyperparam_dict = default_model_gen_dict,
    device = default_device,
    seed = default_seed,
):
    """
    It generates artificially synthetic data using the generative scheme.
    :param starting_data_pack: dict, contains the prior data and other parameters needed to initialize the generative process
    :param model_id: int, id of the trained model to use
    :param hyperparam_dict: dict, contains specifications for the desired synthetic data to be generated
    :param device: torch.device, device to use for the generative process
    :param seed: int, seed for making the synthetic dataset generation reproducible
    :return: dict, contains the generated synthetic data (indicator of states and time steps)
    """

    set_all_seeds(seed)

    model_name = hyperparam_dict['model_name']
    nb_steps = hyperparam_dict['nb_steps']
    delta_t = hyperparam_dict['delta_t']
    input_coords = hyperparam_dict['input_coords']
    output_coords = hyperparam_dict['output_coords']

    # Read the overview file for the model
    df = pd.read_csv(f"{saved_models_path}{model_name}/model_overview.csv", index_col=0)

    # Get the description (parameters) for the desired model_id
    desc = df.loc[df['id'] == model_id, 'description'].values[0]

    # Convert JSON string to dictionary
    params_dict = json.loads(desc)

    # Retriving the main parameters and separating options
    main_params = [
        'input_size',
        'epochs',
        'hidden_size',
        'output_size',
        'bias',
        'ode_nn',
        'readout_nn',
        'enc_nn',
        'use_rnn',
        'dropout_rate',
        'batch_size',
        'solver',
        'dataset',
        'dataset_id',
        'data_dict',
        'learning_rate',
        'test_size',
        'seed',
        'weight',
        'weight_decay',
        'optimal_val_loss'
    ]

    # Separating options from main parameters
    options_params = {k: v for k, v in params_dict.items() if k not in main_params}
    params_model = {k: v for k, v in params_dict.items() if k in main_params}
    params_model['options'] = options_params

    # Retrieve input and output coords
    params_model['input_coords'] = input_coords
    params_model['output_coords'] = output_coords
    params_model['signature_coords'] = input_coords

    params_model['input_size'] = len(input_coords)
    params_model['output_size'] = len(output_coords)

    
    # Selection of best checkpoint based on validation loss
    ckpt_path = f"{saved_models_path}{model_name}/id-{model_id}/best_checkpoint"

    print(f"Using device: {device}")
    # Setting model and optimizer
    model = models.NJODE(**params_model)
    optimizer = torch.optim.Adam(model.parameters())
    models.get_ckpt_model(ckpt_path, model, optimizer, device)
    model.to(device)

    # Running the generation process, built inside the model class
    with torch.no_grad():
        future_path, future_times = model.generate_future_path(
            starting_data_pack=starting_data_pack, steps=nb_steps, delta_t=delta_t,
        )

    # print("Generated state values", future_path.shape, future_path[:, :, :])
    # print("Generated time values:", future_times)

    # Transforminf generated data inot numpy arrays
    # Shape of output process: (nb_paths, nb_steps, output_size)
    # Shape of time steps: (nb_steps,)
    gen_output_process = future_path.detach().cpu().numpy()
    gen_times = future_times
    
    gen_data_dict = {"time_steps": gen_times, "output_process": gen_output_process}

    print("<<< Synthetic data generated successfully >>>")

    return gen_data_dict


def save_gen_synthetic_dataset(
    gen_data_dict,
    model_id=default_model_id,
    prior_data_file_name = default_prior_data_file_name, 
    hyperparam_dict = default_model_gen_dict,
):
    """
    It saves the generated synthetic datasets.
    :param: gen_data_dict: dict, contains the generated data
    :param model_id: int, id of the trained model used to generate the data
    :param prior_data_file_name: str, name of the file which contains prior data
    :param hyperparam_dict: dict, contains specifications for the desired synthetic data to be generated
    :return: str (path where the dataset is saved), int (time_id to identify the dataset)
    """

    # Retriving the generated data
    output_process = gen_data_dict["output_process"]
    time_steps = gen_data_dict["time_steps"]

    # Retriving the dataset overview and adding extra information
    df_overview, data_overview = get_dataset_overview(output_data_gen_process_path)
    model_name = hyperparam_dict['model_name']
    hyperparam_dict['prior_data_file_name'] = prior_data_file_name
    hyperparam_dict['model_id'] = model_id
    original_desc = json.dumps(hyperparam_dict, sort_keys=True)
        

    # time_id = int(time.time())
    time_id = 1
    if len(df_overview) > 0:
        time_id = np.max(df_overview["id"].values) + 1
    file_name = '{}-{}'.format(model_name, time_id)
    path = '{}{}/'.format(output_data_gen_process_path, file_name)

    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError
    df_app = pd.DataFrame(
        data=[[model_name, time_id, original_desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    os.makedirs(path)
    with open('{}gen_output_process.npy'.format(path), 'wb') as f:
        np.save(f, output_process)

    with open('{}gen_time_steps.npy'.format(path), 'wb') as f:
        np.save(f, time_steps)
    
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    print("---- Synthetic dataset saved at path: {} ----".format(path))
    return path, time_id

# =====================================================================================================================


def main(arg):
    """
    function to generate datasets
    """
    del arg

    print("===== Make sure that the prior data is located in the folder: {}{} =====".format(input_data_gen_process_path, '<model_name>'))
    print("===== Verify that the model_id corresponds to a trained model stored in: {}{} =====".format(saved_models_path, '<model_name>/id-<model_id>'))
    print("===== Verify that <model_name> in gen_dataset_params, prior data folder and model folder are the same =====")

    # It creates the folder for the input data if it does not exist
    # (the user has to provide the dataset in this folder)
    makedirs(input_data_gen_process_path)
   
    if FLAGS.model_id:
        model_id = int(FLAGS.model_id)
        print('model_id: {}'.format(model_id))
    else:
        raise ValueError("Please provide --model_id")
        
    if FLAGS.prior_data_file_name:
        prior_data_file_name = FLAGS.prior_data_file_name
        print('prior_data_file_name: {}'.format(prior_data_file_name))
    else:
        raise ValueError("Please provide --prior_data_file_name")
    
    if FLAGS.use_gpu:
        use_gpu = FLAGS.use_gpu
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print('use_gpu: {}'.format(use_gpu))
    else:
        device = default_device
        print('use_gpu: {}'.format(FLAGS.use_gpu))
    
    if FLAGS.gen_dataset_params:
        gen_dataset_params = eval("config."+FLAGS.gen_dataset_params)
        print('gen_dataset_params: {}'.format(gen_dataset_params))
    else:
        raise ValueError("Please provide --gen_dataset_params")
    
    starting_data_pack = preprocessing_data(
    prior_data_file_name = prior_data_file_name,
    hyperparam_dict = gen_dataset_params
    )
    
    gen_data_dict = generative_process(
        starting_data_pack=starting_data_pack,
        model_id = model_id,
        hyperparam_dict = gen_dataset_params,
        device = device,
        seed = FLAGS.gen_seed,
    )

    dataset_path, dataset_id = save_gen_synthetic_dataset(
        gen_data_dict = gen_data_dict,
        model_id = model_id,
        prior_data_file_name = prior_data_file_name, 
        hyperparam_dict = gen_dataset_params
    )
    
if __name__ == '__main__':
    app.run(main)