# IO-NJ-ODE-generative-model-finite-SSP
This repository contains implementations of NJ-ODE framework as a generative model for finite state stochastic processes, as part of a master thesis project.

This work was built upon the original work of Neural Jump ODEs by Dr. Florian Ofenheimer-Krach et al., available at [Neural Jump ODEs](https://github.com/FlorianKrach/PD-NJODE.git) as part of an add-on to the Input Output Neural Jump ODE (IO NJ-ODE) framework to devise an easy and interpretable generative scheme in the context of finite state space processes (SSP).

Two examples of finite SSP were studied for this work:
- Sign of a Brownian motion; referred as "BMClasssification2" withtin the implementation.
- Alternating renewal process; referred as "RPClasssification" withtin the implementation.

Since the goal was to extend capabilities of [Neural Jump ODEs](https://github.com/FlorianKrach/PD-NJODE.git), essentially all files come from there with additions to the following files:

-[models.py](NJODE/models.py)
-[synthetic_datasets.py](NJODE/synthetic_datasets.py)
-[data_utils.py](NJODE/data_utils.py)
-[config.py](NJODE/configs/config.py)
-[config_ParamFilter.py](NJODE/configs/config_ParamFilter.py)

a new script corresponding to the generative scheme:

-[gen_process.py](NJODE/gen_process.py)

and a new folder to conduct the evaluation of the artificially generated paths with such scheme: [analysis](analysis).

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





The configs for the experiment are in the main config file [config.py](NJODE/configs/config.py) as well as in [config_LOB.py](NJODE/configs/config_LOB.py), [config_NJmodel.py](NJODE/configs/config_NJmodel.py) and [config_randomizedNJODE.py](NJODE/configs/config_randomizedNJODE.py).

## Dataset Generation
go to the source directory:
```sh
cd NJODE
```


### Synthetic Datasets

