# IO-NJ-ODE-generative-model-finite-SSP
This repository contains implementations of NJ-ODE framework as a generative model for finite state stochastic processes, as part of a master thesis project.

This work was built upon the original work of Neural Jump ODEs by Dr. Florian Ofenheimer-Krach et al., available at [Neural Jump ODEs](https://github.com/FlorianKrach/PD-NJODE.git) as part of an add-on to the Input Output Neural Jump ODE (IO NJ-ODE) framework to devise an easy and interpretable generative scheme in the context of finite state space processes (SSP).

Two examples of finite SSP were studied for this work:
- Sign of a Brownian motion; referred as "BMClasssification2" withtin the implementation.
- Alternating renewal process; referred as "RPClasssification" withtin the implementation.

Since the goal was to extend capabilities of [Neural Jump ODEs](https://github.com/FlorianKrach/PD-NJODE.git), essentially all files come from there with additions to the following files:

-[models.py](NJODE/models.py); slight modifications in the NJODE class when computing the loss and returning the predicted path of conditional expectation, as well as the addition of the function generate_future_path to artificially generate paths with the described generative scheme.  
-[synthetic_datasets.py](NJODE/synthetic_datasets.py);  BMClasssification2 and RPClassification classes were added to implement forementioned examples. Therein, the implementation of the Monte Carlo estimations for the corresponding conditional expectation can be found, including the particle filtering approach.  
-[data_utils.py](NJODE/data_utils.py); PriorDataset class was added to manage the structure of the input prior data to initialize the gnerative process.
-[config.py](NJODE/configs/config.py); dictionaries for setting the training, evaluation and artificailly generated data of the renewal process, as well as the corresponding model.  
-[config_ParamFilter.py](NJODE/configs/config_ParamFilter.py); dictionaries for setting the training, evaluation and artificailly generated data of the sign of Brownian motion, as well as the corresponding model.  

a new script corresponding to the generative capability:

-[gen_process.py](NJODE/gen_process.py); it manages the loading of the prior input data, the trained model for the generation process and the saving of the generated output data.  

and a new folder to conduct the evaluation of the artificially generated paths: [analysis](analysis).

--------------------------------------------------------------------------------

## Requirements

This code was executed using Python 3.8.10 in VSCode.

To install requirements, download this Repo and cd into it.

Then create a new environment and install all dependencies and this repo.
With [venv]([https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html](https://code.visualstudio.com/docs/python/environments)):
 ```sh
py -3.8 -m venv .venv
Set-ExecutionPolicy Unrestricted -Scope Process # use in case any problem with admin permissions
.\.venv\Scripts\activate
pip install -r requirements_updated.txt
 ```

--------------------------------------------------------------------------------
## Usage, License & Citation

This code can be used in accordance with the [LICENSE](LICENSE).

If you find this code useful or include parts of it in your own work, 
please cite the paper that was mainly used for this work:

- [Nonparametric Filtering, Estimation and Classification using Neural Jump ODEs](https://arxiv.org/abs/2412.03271)
  ```
  @misc{heiss2024nonparametricfilteringestimationclassification,
      title={Nonparametric Filtering, Estimation and Classification using Neural Jump ODEs}, 
      author={Jakob Heiss and Florian Krach and Thorsten Schmidt and Félix B. Tambe-Ndonfack},
      year={2024},
      eprint={2412.03271},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2412.03271}, 
  }
  ```

--------------------------------------------------------------------------------
# Instructions for Running Experiments and Analysis

The configs for the experiment are in the main config file [config.py](NJODE/configs/config.py) as well as in [config_ParamFilter.py](NJODE/configs/config_ParamFilter.py).

Similarly to the flags specified in the original work [Neural Jump ODEs](https://github.com/FlorianKrach/PD-NJODE.git) for the creation of trainig data and the model training:

- **params**: name of the params list (defined in config.py) to use for parallel run
- **NB_JOBS**: nb of parallel jobs to run with joblib
- **first_id**: First id of the given list / to start training of
- **get_overview**: name of the dict (defined in config.py) defining input for extras.get_training_overview
- **USE_GPU**: whether to use GPU for training
- **ANOMALY_DETECTION**: whether to run in torch debug mode
- **SEND**: whether to send results via telegram
- **NB_CPUS**: nb of CPUs used by each training
- **model_ids**: List of model ids to run
- **DEBUG**: whether to run parallel in debug mode
- **saved_models_path**: path where the models are saved
- **overwrite_params**: name of dict (defined in config.py) to use for overwriting params
- **plot_paths**: name of the dict (in config.py) defining input for extras.plot_paths_from_checkpoint
- **climate_crossval**: name of the dict (in config.py) defining input for extras.get_cross_validation
- **plot_conv_study**: name of the dict (in config.py) defining input for extras.plot_convergence_study

--------------------------------------------------------------------------------
## Generative Process
The following flags should be specified for the generative scheme integrated in the script [gen_process.py](NJODE/gen_process.py):

- **gen_dataset_params**: name of the params list (defined in config.py) to use
- **model_id**: id (integer) of trained model to use for generative process
- **prior_data_file_name**: name of the file (<name>.npy) located in data/input_data_generative_process folder, which contains prior data
- **gen_seed*: seed for making the synthetic dataset generation reproducible
- **use_gpu**: whether to use GPU for generative process

It must be said that the prior input data data to initialize the generative process is intended to be provided by the user in a specific folder for this purpose: data/input_data_generative_process after model training, within a corresponding folder whose name corresponds to the model name to be used, i.e., data/input_data_generative_process/BMClassification2 and data/input_data_generative_process/RPClassification. In addition, the prior data must follow the structure of the training data, with the flexibility of being one single observation or a sequence of observations, always starting at time 0. See some of the available examples.

--------------------------------------------------------------------------------
## Dataset Generation
Go to the source directory:
```sh
cd NJODE
```

### Simulations/Real Datasets
Generate datasets:
```shell
py data_utils.py --dataset_name=BMClassification2 --dataset_params=IO_BM2Class2_dict --seed=0
py data_utils.py --dataset_name=BMClassification2 --dataset_params=IO_BMClass2_dict_test --seed=1

py data_utils.py --dataset_name=RPClassification --dataset_params=IO_RPClass_dict --seed=0
py data_utils.py --dataset_name=RPClassification --dataset_params=IO_RPClass_dict_test --seed=1
```

### Training
Train NJODE:
```shell
py run.py --params=param_list_IO_BMClass2 --NB_JOBS=1 --NB_CPUS=1 --get_overview=overview_dict_IO_BMClass2 --USE_GPU=TRUE
py run.py --plot_paths=plot_paths_IO_BMClass2_dict

py run.py --params=param_list_IO_RPClass --NB_JOBS=1 --NB_CPUS=1 --get_overview=overview_dict_IO_RPClass --USE_GPU=TRUE
py run.py --plot_paths=plot_paths_IO_RPClass_dict
```

### Generative Process
Use the generative scheme@
```shell
py gen_process.py --gen_dataset_params=IO_BMClass2_gen_dict --model_id=2 --test1point_bm.npy --seed=277 --use_gpu=True

py gen_process.py --gen_dataset_params=IO_RPClass_gen_dict --model_id=2 --test1point_rp.npy --seed=277 --use_gpu=True
```

### Evaluation
The following files conduct the quality of the artificially generated paths under three different criteria:

- [statistical_fidelity_analysis.py](analysis/statistical_fidelity_analysis.py)
- [discriminative_analysis.py](analysis/discriminative_analysis.py)
- [predictive_analysis.py](analysis/predictive_analysis.py)

These need to be run and modified in a proper code editor according to the needs of each user. The input data must be inside the input_data folder and specify the name of the files to use. More details about usage can be found inside each file.
