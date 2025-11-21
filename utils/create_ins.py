import json
import gym
import torch
from env.case_generator import CaseGenerator

# Generate instances and save to files
def main():
    is_fjsp = False
    batch_size = 20
    num_jobs = 100
    num_mas = 20
    opes_per_job_min = int(num_mas * 0.8) if is_fjsp else 1
    opes_per_job_max = int(num_mas * 1.2) if is_fjsp else 1
    with open("../config.json", 'r') as load_f:
        load_dict = json.load(load_f)

    env_paras = load_dict["env_paras"]
    env_paras["batch_size"] = batch_size
    env_paras["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    problem_type = 'FJSP' if is_fjsp else 'JSSP'
    path = '../data_test/{0}/{1}{2}/'.format(problem_type, num_jobs, str.zfill(str(num_mas),2))
    case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, flag_same_opes=False, flag_doc=True, is_fjsp=is_fjsp, path=path)
    gym.make('fjsp-v0', case=case, env_paras=env_paras)  # Instances are created when the environment is initialized

if __name__ == "__main__":
    main()
    print(f"Instances are created and stored")
