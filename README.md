# 🌈 Unraveling the Rainbow

[![arXiv](https://img.shields.io/badge/arXiv-2505.03323-b31b1b.svg)](https://arxiv.org/abs/2505.03323)

This is the repository of the paper **_Unraveling the Rainbow: can value-based methods schedule?_**

---

## 📦 Dependencies

Make sure you have the following installed:

| Package  | Version |
|----------|---------|
| 🐍 Python   | ≥ 3.11.11 |
| 🔥 PyTorch  | ≥ 2.4.0 |
| 🎮 Gym      | ≥ 0.22.0 |
| 📊 NumPy    | ≥ 2.0.2 |
| 🐼 Pandas   | ≥ 2.2.3 |

The repository is organized as follows:

- **`/data_dev`** and **`/data_test`**: Validation and testing instances for JSSP and FJSP of various sizes.  
- **`/env`**: Implementation of the scheduling environment.  
- **`/models`**: Models for value-based and policy-gradient algorithms.  
- **`/network`**: Utilities for heterogeneous GNN, MLPs, and noisy linear layers.  
- **`/results`**: Results for each individual benchmark instance.  
- **`/save`**: Stores trained model parameters.  
- **`/utils`**: Utility scripts for training and testing.  
- **`config.json`**: Central configuration for experiments.  
- **Training scripts:**  
  - `train_dqn.py` → Value-based algorithms  
  - `train_a2c.py` → A2C  
  - `train_ppo.py` → PPO  
  - `train_reinforce.py` → REINFORCE  
  - `train_vmpo.py` → V-MPO  
- **Validation/testing scripts:**  
  - `validate.py` → Validation steps  
  - `test.py` → Randomly generated instances  
  - `test_benchmark.py` → Benchmark instances

---

## ⚙️ Running Experiments

## 1. 🏋️ Training a Model

To train a model, you first need to configure your experiment in `config.json`. This includes problem type, instance size, algorithm, and any Rainbow extensions (for value-based methods).

---


### 🔹 Problem Type

Set the `env_paras.is_fjsp` flag:

| Flag | Problem |
|------|---------|
| `true`  | FJSP (Flexible Job Shop Problem) |
| `false` | JSSP (Job Shop Scheduling Problem) |


### 🔹 Instance size

Configure the number of jobs and machines in `env_paras`, e.g. (20 jobs and 10 machines):

```
"env_paras": {
  "num_jobs": 20,
  "num_mas": 10
}
```

### 🔹 Running Training

Value-based algorithms (DQN and Rainbow configurations): 
```
python train_dqn.py
```

Policy-gradient algorithms: 
```
python train_ppo.py       # PPO
python train_a2c.py       # A2C
python train_reinforce.py # REINFORCE
python train_vmpo.py      # V-MPO
```

### 🔹 Results naming

The ```config_name``` field determines the save path.
Example:
```
"config_name": "D3QN"
```
Results are saved to:
```
save/<problem_type>/<instance_size>/train_D3QN_<instance_size>
```

### 🔹 Rainbow extensions:

Enable optional Rainbow extensions by setting Boolean flags in `extensions_paras` (e.g., `"use_noisy": true`). This should only be set when using value-based algorithms.

### 🔹 Notes:

* Training scripts automatically create the folder structure and save model parameters.

* ```save_output``` in ```config.json``` controls whether outputs are saved.

* ```load_state``` and ```load_epoch``` can be set if you want to resume training from a checkpoint.

## 2. 🧪 Testing a Model

Testing a model is similar to training in terms of configuration. You need to set the environment, algorithm, and instance size properly.

---

### 🔹 Environment

Set the `is_fjsp` flag in `env_paras` to choose the problem type, as well as `num_jobs` and `num_machines` for the instance size.

### 🔹 Test Parameters (`test_paras`)

Configure the `test_paras` section in `config.json`. Example:

```
"test_paras": {
  "sample": false,
  "pomo_starting_nodes": false,
  "topk": 5,
  "is_ppo": false,
  "is_a2c": false,
  "is_reinforce": false,
  "is_vmpo": false,
  "is_sql": false,
  "num_sample": 25,
  "saved_model_num_jobs": 6,
  "saved_model_num_mas": 6,
  "benchmark_path": "data_test/FJSP/Public/Brandimarte/"
}
```

### Key Parameters Explained

| Parameter | Description                                                                                                                                                    |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sample` | If `true`, uses sampling instead of greedy decoding.                                                                                                           |
| `topk` | Number of top actions to consider (only for sampling).                                                                                                         |
| `pomo_starting_nodes` | If `true`, uses POMO multistart decoding.                                                                                                                      |
| `max_pomo_nodes` | Number of maximum POMO starting nodes. Set it as `null` if you want to use the maximum number of nodes possible. |
| `is_ppo`, `is_a2c`, `is_reinforce`, `is_vmpo`| Set `true` for the algorithm you used during training and false for the rest.                                                                                  |
| `num_sample` | Number of random samples per instance (used for sampling).                                                                                                     |
| `saved_model_num_jobs`, `saved_model_num_mas` | Instance size used during training, required to load the correct model.                                                                                        |
| `benchmark_path` | Path to benchmark dataset for evaluation.                                                                                                                      |

### 🔹 Running tests

For randomly generated problems, run:

```
python test.py
```

For benchmark instances, run:

```
python test_benchmark.py
```

## 3. 📊 Reproducing Paper Results

To reproduce the results from our paper, simply run ```python reproduce_scripts/test.py``` and ```python reproduce_scripts/test_benchmark.py``` as they are.

## 🙏 Acknowledgments

We would like to thank the following repositories, which served as important foundations for our work:

* https://github.com/songwenas12/fjsp-drl/tree/main
* https://github.com/Curt-Park/rainbow-is-all-you-need

## 📖 Citation

If you find our work valuable for your research, please cite us:

```
@misc{corrêa2025unravelingrainbowvaluebasedmethods,
      title={Unraveling the Rainbow: can value-based methods schedule?}, 
      author={Arthur Corrêa and Alexandre Jesus and Cristóvão Silva and Paulo Nascimento and Samuel Moniz},
      year={2025},
      eprint={2505.03323},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.03323}, 
}
```
