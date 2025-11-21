import copy
import json
import os
import random
import time as time

import gym
import pandas as pd
import torch
import numpy as np

from models import dqn_model, PPO_model, A2C_model, REINFORCE_model, VMPO_model
from env.load_data import nums_detec

from utils.my_utils import pomo_starting_nodes


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(env_paras, model_paras, train_paras, extension_paras, test_paras, config_name=None, instance_size=None,
         results=None):
    device = torch.device("cuda:0")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None,
                           sci_mode=False)

    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)

    env_paras["device"] = device
    model_paras["device"] = device
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    problem_type = 'FJSP' if env_paras["is_fjsp"] else 'JSSP'

    assert (extension_paras["use_iqn"] + extension_paras["use_distributional"]) <= 1
    assert (extension_paras["use_munchausen"] + extension_paras["use_distributional"]) <= 1
    assert (extension_paras["use_munchausen"] + extension_paras["use_ddqn"]) <= 1

    seed = train_paras["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    data_path = test_paras["benchmark_path"]
    test_files = sorted(os.listdir(data_path))  # Alphabetical order
    test_files = [os.path.join(data_path, f) for f in test_files]
    if "JSSP/Public" in test_paras["benchmark_path"]:
        is_public_jssp = True
    else:
        is_public_jssp = False
    env_test_paras["batch_size"] = 1

    if test_paras["is_ppo"]:
        model = PPO_model.PPO(model_paras, train_paras)
    elif test_paras["is_a2c"]:
        model = A2C_model.A2C(model_paras, train_paras)
    elif test_paras["is_reinforce"]:
        model = REINFORCE_model.REINFORCE(model_paras, train_paras)
    elif test_paras["is_vmpo"]:
        model = VMPO_model.VMPO(model_paras, train_paras)
    else:
        model = dqn_model.Model(model_paras, train_paras, extension_paras, test_paras["topk"])

    save_path = './save/{0}/{1}{2}/train_{3}_{4} x {5}'.format(problem_type, test_paras["saved_model_num_jobs"],
                                                               str.zfill(str(test_paras["saved_model_num_mas"]), 2),
                                                               train_paras["config_name"],
                                                               test_paras["saved_model_num_jobs"],
                                                               test_paras["saved_model_num_mas"])
    files = os.listdir(save_path)
    base_filename = 'save_best_{0}_{1} x {2}_'.format(train_paras["config_name"], test_paras["saved_model_num_jobs"],
                                                      test_paras["saved_model_num_mas"])
    best_model_match = None
    for file in files:
        if file.startswith(base_filename) and file.endswith('.pt'):
            best_model_match = file
            break
    checkpoint_best_model = torch.load(os.path.join(save_path, best_model_match), map_location=device)
    if test_paras["is_ppo"]:
        model.policy_old.load_state_dict(checkpoint_best_model)
    elif test_paras["is_a2c"]:
        model.policy.load_state_dict(checkpoint_best_model)
    elif test_paras["is_reinforce"]:
        model.policy.load_state_dict(checkpoint_best_model)
    elif test_paras["is_vmpo"]:
        model.policy_old.load_state_dict(checkpoint_best_model)
    else:
        model.online_network.load_state_dict(checkpoint_best_model)

    start = time.time()
    print('There are {0} dev instances.'.format(len(test_files)))  # validation set is also called development set
    total_makespan = 0

    if results != None:
        result_column_name = f"{instance_size} - {config_name}"

        if result_column_name not in results:
            results[result_column_name] = []

    if not test_paras["sample"]:
        for instance in test_files:
            if test_paras["pomo_starting_nodes"]:
                initial_actions, env = pomo_starting_nodes(instance, copy.deepcopy(env_test_paras), is_public_jssp)
                env.reset()
                first_timestep = True
            else:
                env = gym.make('fjsp-v0', case=instance, env_paras=env_test_paras, data_source='public',
                               is_public_jssp=is_public_jssp)
                env.reset()

            state = env.state
            done = False
            dones = env.done_batch
            while ~done:
                with torch.no_grad():
                    if test_paras["pomo_starting_nodes"] and first_timestep:
                        first_timestep = False
                        actions = initial_actions
                    else:
                        if test_paras["is_ppo"]:
                            actions = model.policy_old.act(state, None, dones, epsilon=0, flag_sample=False,
                                                           flag_train=False)
                        elif test_paras["is_a2c"]:
                            actions = model.policy.act(state, None, dones, epsilon=0, flag_sample=False,
                                                       flag_train=False)
                        elif test_paras["is_reinforce"]:
                            actions = model.policy.act(state, None, dones, epsilon=0, flag_sample=False,
                                                       flag_train=False)
                        elif test_paras["is_vmpo"]:
                            actions = model.policy_old.act(state, None, dones, epsilon=0, flag_sample=False,
                                                           flag_train=False)
                        else:
                            actions = model.online_network.act(state, None, dones, epsilon=0, flag_sample=False,
                                                               flag_train=False)
                state, rewards, dones, _ = env.step(actions)
                done = dones.all()

            # gantt_result = env.validate_gantt()[0]
            # if not gantt_result:
            #    print("Scheduling Error！！！！！！")

            makespan = copy.deepcopy(env.makespan_batch.min()).cpu().item()
            total_makespan += makespan
            if results != None:
                results[result_column_name].append(makespan)

            print('testing time: ', round(time.time() - start, 2), ' | makespan: ', round(makespan, 2), ' | instance: ',
                  instance.rsplit("/", 1)[-1].split(".", 1)[0])
        print('\n', 'average makespan: ', round(total_makespan / len(test_files), 2))
    else:
        for instance in test_files:
            if test_paras["pomo_starting_nodes"]:
                initial_actions, env = pomo_starting_nodes(instance, copy.deepcopy(env_test_paras), is_public_jssp,
                                                           samples=test_paras["num_sample"])
                env.reset()
                first_timestep = True
            else:
                S = test_paras["num_sample"]
                env_test_paras["batch_size"] = 1 * S
                env = gym.make('fjsp-v0', case=instance, env_paras=env_test_paras, data_source='public',
                               is_public_jssp=is_public_jssp)
                env.reset()

            state = env.state
            done = False
            dones = env.done_batch
            while ~done:
                with torch.no_grad():
                    if test_paras["pomo_starting_nodes"] and first_timestep:
                        first_timestep = False
                        actions = initial_actions
                    else:
                        if test_paras["is_ppo"]:
                            actions = model.policy_old.act(state, None, dones, epsilon=0, flag_sample=True,
                                                           flag_train=False)
                        elif test_paras["is_a2c"]:
                            actions = model.policy.act(state, None, dones, epsilon=0, flag_sample=True, flag_train=False)
                        elif test_paras["is_reinforce"]:
                            actions = model.policy.act(state, None, dones, epsilon=0, flag_sample=True, flag_train=False)
                        elif test_paras["is_vmpo"]:
                            actions = model.policy_old.act(state, None, dones, epsilon=0, flag_sample=True,
                                                           flag_train=False)
                        else:
                            actions = model.online_network.act(state, None, dones, epsilon=0, flag_sample=True,
                                                               flag_train=False)
                state, rewards, dones, _ = env.step(actions)
                done = dones.all()

            # gantt_result = env.validate_gantt()[0]
            # if not gantt_result:
            #    print("Scheduling Error！！！！！！")

            makespans = copy.deepcopy(env.makespan_batch).cpu().numpy()  # shape: (S,)
            min_makespan = makespans.min()
            total_makespan += min_makespan
            if results != None:
                results[result_column_name].append(min_makespan)

            print('testing time: ', round(time.time() - start, 2), ' | makespan: ', round(min_makespan, 2),
                  ' | instance: ', instance.rsplit("/", 1)[-1].split(".", 1)[0])
        print('\n', 'average makespan: ', round(total_makespan / len(test_files), 2))


if __name__ == '__main__':
    instance_sizes = [(6, 6)]
    results = {}

    for size in instance_sizes:
        config_names = ["DQN", "DDQN", "PER", "Dueling", "Noisy", "Distributional", "NStep", "Rainbow", "PPO", "A2C",
                        "REINFORCE", "VMPO"]
        config_uses = [[False, False, False, False, False, False],
                       [True, False, False, False, False, False],
                       [False, True, False, False, False, False],
                       [False, False, True, False, False, False],
                       [False, False, False, True, False, False],
                       [False, False, False, False, True, False],
                       [False, False, False, False, False, True],
                       [True, True, True, True, True, True]]

        for config in range(12, 13):
            print("#####################################################################################")
            print(f"Running {config_names[config]} model - Instance size: {size[0]}x{size[1]}")

            # Load config and init objects
            with open("./config.json", 'r') as load_f:
                load_dict = json.load(load_f)
            env_paras = copy.deepcopy(load_dict["env_paras"])
            model_paras = copy.deepcopy(load_dict["model_paras"])

            extension_paras = copy.deepcopy(load_dict["extensions_paras"])
            test_paras = copy.deepcopy(load_dict["test_paras"])
            test_paras["saved_model_num_jobs"] = size[0]
            test_paras["saved_model_num_mas"] = size[1]

            if config_names[config] == "PPO":
                train_paras = copy.deepcopy(load_dict["ppo_paras"])
                test_paras["is_ppo"] = True
            elif config_names[config] == "A2C":
                train_paras = copy.deepcopy(load_dict["a2c_paras"])
                test_paras["is_a2c"] = True
            elif config_names[config] == "REINFORCE":
                train_paras = copy.deepcopy(load_dict["reinforce_paras"])
                test_paras["is_reinforce"] = True
            elif config_names[config] == "VMPO":
                train_paras = copy.deepcopy(load_dict["vmpo_paras"])
                test_paras["is_vmpo"] = True
            else:
                train_paras = copy.deepcopy(load_dict["dqn_paras"])
                extension_paras["use_ddqn"] = config_uses[config][0]
                extension_paras["use_per"] = config_uses[config][1]
                extension_paras["use_dueling"] = config_uses[config][2]
                extension_paras["use_noisy"] = config_uses[config][3]
                extension_paras["use_distributional"] = config_uses[config][4]
                extension_paras["use_n_step"] = config_uses[config][5]

            train_paras["config_name"] = config_names[config]
            main(env_paras, model_paras, train_paras, extension_paras, test_paras, config_names[config], size, results)

    df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in results.items()]))

    # Save results to an Excel file
    output_file = f"results/greedy/results_brandimarte.xlsx"
    df.to_excel(output_file, index=False)
