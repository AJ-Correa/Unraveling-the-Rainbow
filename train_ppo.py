import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

from models import PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env

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


def main(env_paras, model_paras, train_paras):
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    problem_type = 'FJSP' if env_paras["is_fjsp"] else 'JSSP'

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

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
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
        model.policy.load_state_dict(checkpoint)
        model.policy_old.load_state_dict(checkpoint)
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
            best_ppo_model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
            best_ppo_model.policy_old.load_state_dict(checkpoint_best_model)
            makespan_best, _ = validate(env_valid_paras, env_valid, best_ppo_model.policy_old)
            del best_ppo_model

    # Start training iteration
    start_time = time.time()
    env = None

    for i in range(start_epoch, train_paras["max_iterations"] + 1):
        # Replace training instances every x iteration
        if (i - 1) % train_paras["parallel_iter"] == 0 or env is None:
            if env_paras["is_fjsp"]:
                nums_ope = generate_fixed_sum_list(num_jobs, num_jobs * num_mas, opes_per_job_min, opes_per_job_max)
            else:
                nums_ope = [num_mas for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope, is_fjsp=env_paras["is_fjsp"])
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)

        env.reset()

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones, _ = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        env.reset()

        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print("epoch: ", i, " | spend_time: ", round(time.time() - last_time, 2), " | reward: ", round(reward, 2), " | loss: ", round(loss, 2))
            memories.clear_memory()

        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.policy_old)

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
                    torch.save(model.policy.state_dict(), save_best_model)

                if len(current_model) == 1:
                    delete_current_model = current_model.popleft()
                    os.remove(delete_current_model)
                save_current_model = '{0}/save_last_{1}_{2}_{3}.pt'.format(save_path, train_paras["config_name"], f"{num_jobs} x {num_mas}", i)
                current_model.append(save_current_model)
                torch.save(model.policy.state_dict(), save_current_model)

    print("total_time: ", round((time.time() - start_time) / 60, 2))

if __name__ == '__main__':
    config_name = "PPO"
    # instance_sizes = [(6, 6), (10, 5), (20, 5), (15, 10), (20, 10)]
    instance_sizes = [(20, 10)]

    problem_types = ["JSSP"]

    for problem in problem_types:
        for size in instance_sizes:
            with open("./config.json", 'r') as load_f:
                load_dict = json.load(load_f)
            env_paras = copy.deepcopy(load_dict["env_paras"])
            model_paras = copy.deepcopy(load_dict["model_paras"])
            train_paras = copy.deepcopy(load_dict["ppo_paras"])
            train_paras["config_name"] = config_name
            env_paras["is_fjsp"] = True if problem == "FJSP" else False
            env_paras["num_jobs"] = size[0]
            env_paras["num_mas"] = size[1]

            main(env_paras, model_paras, train_paras)
