# Unraveling the Rainbow

[![arXiv](https://img.shields.io/badge/arXiv-2505.03323-b31b1b.svg)](https://arxiv.org/abs/2505.03323)

Implementation of the paper **Unraveling the Rainbow: can value-based methods schedule?**

### Dependencies

* python $\ge$ 3.11.11
* pytorch $\ge$ 2.4.0
* gym $\ge$ 0.22.0
* numpy $\ge$ 2.0.2
* pandas $\ge$ 2.2.3

### Introduction

* ```/data_dev``` and ```/data_test``` are the directories storing the validation and testing instances for both the job-shop (JSSP) and flexible job-shop scheduling (FJSP) problems, of all different sizes considered in our work.
* ```/env``` contains the implementation of the problem environment.
* ```/models``` contains the implementation of the models for the value-based algorithms and PPO.
* ```/network``` contains utilities code for the heterogeneous GNN model, multi-layer perceptron and noisy linear layers.
* ```/results``` contains the results on each individual benchmark instance used in our paper.
* ```/save``` is the directory used to store all different models trained.
* ```/utils``` contains different utilities used during training and testing.
* ```config.json``` is the configuration file, which has all different parameters that must be adjusted for different scenarios.
* ```PPO_model.py``` contains the implementation of the algorithms in this article, including HGNN and PPO algorithms
* ```test.py``` for testing
* ```train_dqn.py``` is the training file for value-based algorithms.
* ```train_ppo.py``` is the training file for the PPO algorithm.
* ```validate.py``` is the file used for performing the validation steps.
* ```test.py``` is the testing file for randomly generated instances.
* ```test_benchmark.py``` is the testing file for benchmark instances.

## Reproducing the results from the paper

In our paper, we perform multiple different experiments on both the JSSP and FJSP, covering multiple instance sizes and algorithms. 
To run each one, change the ```config.json``` file accordingly, specifically the following parameters:

* **Problem type:** set ```is_fjsp``` in ```env_paras``` to ```true``` for FJSP experiments, or ```false``` for JSSP experiments.
* **Instance size:** adjust ```num_jobs``` and ```num_mas``` in ```env_paras``` to match the problem instance (e.g., 20 jobs and 10 machines).
* **Algorithm selection:** use the ```config_name``` field in either ```dqn_paras``` or ```ppo_paras``` to select the desired algorithm (e.g., ```"DQN"```, ```"PPO"```).
If using any Rainbow extensions, set the boolean parameters accordingly in ```extensions_paras``` (e.g., ```"use_noisy": true```).
* **Testing:** for evaluation, configure ```test_paras``` appropriately (e.g., set ```"is_ppo": true``` if using PPO, specify the instance size used during training by setting ```saved_model_num_jobs``` and ```saved_model_num_mas``` accordingly)

### Training

For training any value-based algorithm, after setting the correct parameters in ```config.json```, run the following command:

```
python train_dqn.py
```

For the PPO algorithm, run:

```
python train_ppo.py
```

### Testing

For testing, just set the boolean variable ```is_fjsp``` in ```env_paras``` accordingly.

For randomly generated problems, run:

```
python test.py
```

For benchmark instances, do not forget to set the proper path (```benchmark_path``` in ```test_paras```) for the benchmark dataset to be used for evaluation. To properly save the results on each individual instance on an excel, change the ```output_path``` variable name as needed, at the end of the ```test_benchmark.py``` file. After, run:

```
python test_benchmark.py
```

## Acknowledgments

We would like to thank the following repositories, which served as important foundations for our work:

* https://github.com/songwenas12/fjsp-drl/tree/main
* https://github.com/Curt-Park/rainbow-is-all-you-need

## Citation

If you find our work valuable for your research, please cite us:

```
@misc{corrêa2025unravelingrainbowvaluebasedmethods,
      title={Unraveling the Rainbow: can value-based methods schedule?}, 
      author={Arthur Corrêa and Alexandre Jesus and Cristóvão Silva and Samuel Moniz},
      year={2025},
      eprint={2505.03323},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.03323}, 
}
```
