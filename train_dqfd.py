import copy
import json
import os
import random
import math
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
# from visdom import Visdom

from models import dqn_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env
from utils.replay_memory import ReplayBuffer, PrioritizedReplayBuffer
from or_tools.or_tools import flexible_jobshop


def generate_fixed_sum_list(num_jobs, total_sum, min_val, max_val):
    nums_ope = [random.randint(min_val, max_val) for _ in range(num_jobs)]

    # Step 2: Adjust the sum
    current_sum = sum(nums_ope)
    difference = current_sum - total_sum

    # Step 3: Adjust elements to ensure the sum equals the target
    while difference != 0:
        for i in range(num_jobs):
            if difference > 0 and nums_ope[i] > min_val:
                adjustment = min(difference, nums_ope[i] - min_val)
                nums_ope[i] -= adjustment
                difference -= adjustment
            elif difference < 0 and nums_ope[i] < max_val:
                adjustment = min(-difference, max_val - nums_ope[i])
                nums_ope[i] += adjustment
                difference += adjustment

            if difference == 0:
                break

    return nums_ope


def main(env_paras, model_paras, train_paras, extension_paras):
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    env_paras["batch_size"] = 1
    problem_type = 'FJSP' if env_paras["is_fjsp"] else 'JSSP'

    assert (extension_paras["use_iqn"] + extension_paras["use_distributional"]) <= 1
    assert (extension_paras["use_munchausen"] + extension_paras["use_distributional"]) <= 1
    assert (extension_paras["use_munchausen"] + extension_paras["use_ddqn"]) <= 1
    extension_paras["use_per"] = 1

    seed = train_paras["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.8) if env_paras["is_fjsp"] else 1
    opes_per_job_max = int(num_mas * 1.2) if env_paras["is_fjsp"] else 1

    extension_paras["n_step_horizon"] = 1
    memories = PrioritizedReplayBuffer(max_size=train_paras["buffer_size"], batch_size=train_paras["minibatch_size"],
                                       alpha=extension_paras["per_alpha"], beta=extension_paras["per_beta"],
                                       gamma=train_paras["gamma"], n_step_horizon=extension_paras["n_step_horizon"], use_n_step=extension_paras["use_n_step"], to_cpu=model_paras["replay_to_cpu"])

    model = dqn_model.Model(model_paras, train_paras, extension_paras)
    start_epoch = 1
    best_model = deque()
    current_model = deque()
    makespan_best = float('inf')

    if model_paras["load_state"]:
        checkpoint_fullname = './save/{0}/{1}{2}/train_{3}_{4} x {5}/save_last_{6}_{7} x {8}_{9}.pt'.format(problem_type, num_jobs,
                                                                                      str.zfill(str(num_mas),2),
                                                                                      train_paras["config_name"],
                                                                                      num_jobs, num_mas,
                                                                                      train_paras["config_name"],
                                                                                      num_jobs, num_mas,
                                                                                      model_paras["load_epoch"])
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        model.online_network.load_state_dict(checkpoint)
        model.target_network.load_state_dict(checkpoint)
        start_epoch = 1 + model_paras["load_epoch"]

    env_valid = get_validate_env(env_valid_paras)  # Create an environment for validation

    if train_paras["save_output"]:
        # Generate data files and fill in the header
        save_path = './save/{0}/{1}{2}/train_{3}_{4}'.format(problem_type, env_paras["num_jobs"], str.zfill(str(env_paras["num_mas"]),2), train_paras["config_name"], f"{num_jobs} x {num_mas}")
        os.makedirs(save_path, exist_ok=True)

        valid_results = []
        valid_results_100 = []

        if not os.path.exists(f'{save_path}/training_ave.xlsx'):
            pd.DataFrame(columns=["epochs", "res"]).to_excel(f'{save_path}/training_ave.xlsx', index=False)

        if not os.path.exists(f'{save_path}/training_100.xlsx'):
            pd.DataFrame(columns=["epochs"] + [f"instance_{i}" for i in range(100)]).to_excel(
                f'{save_path}/training_100.xlsx', index=False)

        if model_paras["load_state"]:
            files = os.listdir(save_path)

            # Generate the base part of the filename without the dynamic element (e.g., without 'i')
            base_filename = 'save_best_{0}_{1}_'.format(train_paras["config_name"], f"{num_jobs} x {num_mas}")

            # Look for the exact match in the list of files
            best_model_match = None
            for file in files:
                if file.startswith(base_filename) and file.endswith('.pt'):
                    best_model_match = file
                    break
            best_model.append(os.path.join(save_path, best_model_match))
            current_model.append('{0}/save_last_{1}_{2}_{3}.pt'.format(save_path,
                                                                       train_paras["config_name"], f"{num_jobs} x {num_mas}", start_epoch - 1))

            checkpoint_best_model = torch.load(os.path.join(save_path, best_model_match), map_location=device)
            best_dqn_model = dqn_model.Model(model_paras, train_paras, extension_paras)
            best_dqn_model.online_network.load_state_dict(checkpoint_best_model)
            makespan_best, _ = validate(env_valid_paras, env_valid, best_dqn_model.online_network)
            del best_dqn_model

    # Start training iteration
    start_time = time.time()
    env = None
    max_epsilon = train_paras["max_epsilon"]
    min_epsilon = train_paras["min_epsilon"]
    epsilon_decay = train_paras["eps_decay"]
    epsilon = max_epsilon if not extension_paras["use_noisy"] else 0
    if model_paras["load_state"]:
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(
                    -1. * start_epoch / epsilon_decay)

    for inst in range(train_paras["imitation_instances"]):
        if env_paras["is_fjsp"]:
            nums_ope = generate_fixed_sum_list(num_jobs, num_jobs * num_mas, opes_per_job_min, opes_per_job_max)
        else:
            nums_ope = [num_mas for _ in range(num_jobs)]
        case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope, is_fjsp=env_paras["is_fjsp"])
        env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
        # print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))
        _, _, labels = flexible_jobshop(env.lines[0], 3)
        env.reset()

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()
        episode_reward = 0

        for transition in labels:
            model.online_network.add_transition(state)
            action_indexes = torch.tensor(transition[1] * num_jobs + transition[2])
            model.online_network.transition += (action_indexes,)
            actions = torch.tensor([[transition[0]], [transition[1]], [transition[2]]])
            print(actions)
            state, rewards, dones, _ = env.step(actions)
            episode_reward += rewards.item()

            model.online_network.add_next_state(state, memories)

            model.online_network.transition += (rewards,
                                                dones)
            one_step_transition = model.online_network.transition

            if one_step_transition:
                memories.store(one_step_transition)

            if extension_paras["use_per"]:
                memories.sum_tree[memories.tree_ptr] = memories.max_priority ** memories.alpha
                memories.min_tree[memories.tree_ptr] = memories.max_priority ** memories.alpha
                memories.tree_ptr = (memories.tree_ptr + 1) % memories.max_size

            #if len(memories.ope_ma_adj) >= train_paras["minibatch_size"]:
            #    model.update_model(memories)
        print(env.makespan_batch)
        print(labels[-1][-1])
        print()

    for i in range(start_epoch, train_paras["epochs"] + 1):
        # Replace training instances every x epochs
        if (i - 1) % train_paras["switch_instance"] == 0:
            if env_paras["is_fjsp"]:
                nums_ope = generate_fixed_sum_list(num_jobs, num_jobs * num_mas, opes_per_job_min, opes_per_job_max)
            else:
                nums_ope = [num_mas for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope, is_fjsp=env_paras["is_fjsp"])
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            # print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        env.reset()

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()
        episode_reward = 0

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.online_network.act(state, memories, dones, epsilon)
            state, rewards, dones, _ = env.step(actions)
            done = dones.all()
            episode_reward += rewards.item()

            # memories.rewards.append(rewards)
            # memories.is_terminals.append(dones)

            model.online_network.add_next_state(state, memories)

            model.online_network.transition += (rewards,
                                                dones)
            one_step_transition = model.online_network.transition

            if one_step_transition:
                memories.store(one_step_transition)

            if extension_paras["use_per"]:
                memories.sum_tree[memories.tree_ptr] = memories.max_priority ** memories.alpha
                memories.min_tree[memories.tree_ptr] = memories.max_priority ** memories.alpha
                memories.tree_ptr = (memories.tree_ptr + 1) % memories.max_size

            if not extension_paras["use_noisy"]:
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(
                    -1. * i / epsilon_decay)
                # epsilon = max(
                #         min_epsilon, epsilon - (
                #             max_epsilon - min_epsilon
                #         ) * epsilon_decay
                #     )

            if len(memories.ope_ma_adj) >= train_paras["minibatch_size"]:
                model.update_model(memories)

            # gpu_tracker.track()  # Used to monitor memory (of gpu)

        if i % train_paras["target_update"] == 0:
            model.target_network.load_state_dict(model.online_network.state_dict())

        if extension_paras["use_per"]:
            fraction = min(i / train_paras["epochs"], 1.0)
            memories.beta = memories.beta + fraction * (1.0 - memories.beta)

        print("epoch: ", i, " | spend_time: ", round(time.time() - last_time, 2), " | makespan: ", env.makespan_batch.item(), " | reward: ", round(episode_reward, 2), " | epsilon: ", round(epsilon, 2))

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env.reset()

        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        if i % train_paras["target_update"] == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.online_network)

            if train_paras["save_output"]:
                valid_results.append([i, vali_result.item()])
                valid_results_100.append([i] + vali_result_100.tolist())

                # **Append validation results to the Excel file**
                with pd.ExcelWriter(f'{save_path}/training_ave.xlsx', mode='a', engine='openpyxl',
                                    if_sheet_exists='overlay') as writer:
                    pd.DataFrame(valid_results).to_excel(writer, sheet_name='Sheet1', index=False, header=False,
                                                         startrow=writer.sheets['Sheet1'].max_row)

                with pd.ExcelWriter(f'{save_path}/training_100.xlsx', mode='a', engine='openpyxl',
                                    if_sheet_exists='overlay') as writer:
                    pd.DataFrame(valid_results_100).to_excel(writer, sheet_name='Sheet1', index=False, header=False,
                                                             startrow=writer.sheets['Sheet1'].max_row)

                valid_results.clear()
                valid_results_100.clear()

                # Save the best model
                if vali_result < makespan_best:
                    makespan_best = vali_result
                    if len(best_model) == 1:
                        delete_best_model = best_model.popleft()
                        os.remove(delete_best_model)
                    save_best_model = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, train_paras["config_name"], f"{num_jobs} x {num_mas}", i)
                    best_model.append(save_best_model)
                    torch.save(model.online_network.state_dict(), save_best_model)

                if len(current_model) == 1:
                    delete_current_model = current_model.popleft()
                    os.remove(delete_current_model)
                save_current_model = '{0}/save_last_{1}_{2}_{3}.pt'.format(save_path, train_paras["config_name"], f"{num_jobs} x {num_mas}", i)
                current_model.append(save_current_model)
                torch.save(model.online_network.state_dict(), save_current_model)

    print("total_time: ", round((time.time() - start_time) / 60, 2))

if __name__ == '__main__':

    # config_names = ["DQN", "DDQN", "PER", "Dueling", "Noisy", "Distributional", "NStep", "Rainbow"]
    # config_uses = [[False, False, False, False, False, False],
    #                [True, False, False, False, False, False],
    #                [False, True, False, False, False, False],
    #                [False, False, True, False, False, False],
    #                [False, False, False, True, False, False],
    #                [False, False, False, False, True, False],
    #                [False, False, False, False, False, True],
    #                [True, True, True, True, True, True]]
    #
    # for config in range(8):
    #     # Load config and init objects
    #     with open("./config.json", 'r') as load_f:
    #         load_dict = json.load(load_f)
    #     env_paras = copy.deepcopy(load_dict["env_paras"])
    #     model_paras = copy.deepcopy(load_dict["model_paras"])
    #     train_paras = copy.deepcopy(load_dict["dqn_paras"])
    #     extension_paras = copy.deepcopy(load_dict["extensions_paras"])
    #
    #     train_paras["config_name"] = config_names[config]
    #     extension_paras["use_ddqn"] = config_uses[config][0]
    #     extension_paras["use_per"] = config_uses[config][1]
    #     extension_paras["use_dueling"] = config_uses[config][2]
    #     extension_paras["use_noisy"] = config_uses[config][3]
    #     extension_paras["use_distributional"] = config_uses[config][4]
    #     extension_paras["use_n_step"] = config_uses[config][5]
    #
    #     main(env_paras, model_paras, train_paras, extension_paras)

    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = copy.deepcopy(load_dict["env_paras"])
    model_paras = copy.deepcopy(load_dict["model_paras"])
    train_paras = copy.deepcopy(load_dict["dqn_paras"])
    extension_paras = copy.deepcopy(load_dict["extensions_paras"])

    main(env_paras, model_paras, train_paras, extension_paras)
