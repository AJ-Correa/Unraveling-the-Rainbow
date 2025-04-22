import gym
import env
import torch
import time
import os
import copy

def get_validate_env(env_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    problem_type = "FJSP" if env_paras["is_fjsp"] else "JSSP"
    file_path = "./data_dev/{0}/{1}{2}/".format(problem_type, env_paras["num_jobs"], str.zfill(str(env_paras["num_mas"]),2))
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=env_paras, data_source='file')
    return env

def validate(env_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set

    env.reset()
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, None, dones, epsilon=0, flag_sample=False, flag_train=False)
        state, rewards, dones, _ = env.step(actions)
        done = dones.all()
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    env.reset()
    print('validating time: ', round(time.time() - start, 2), ' | average makespan: ', round(makespan.item(), 2), '\n')
    return makespan, makespan_batch
