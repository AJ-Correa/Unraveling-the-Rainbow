import torch
import json
import gym
import copy


def read_json(path: str) -> dict:
    with open(path + ".json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def write_json(data: dict, path: str):
    with open(path + ".json", 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(data, indent=2, ensure_ascii=False))


def benchmark_pomo_starting_nodes(instance, env_test_paras, is_public_jssp, samples=None, max_starting_nodes=None):
    env_test_paras["batch_size"] = 1
    env = gym.make('fjsp-v0', case=instance, env_paras=env_test_paras, data_source='public',
                   is_public_jssp=is_public_jssp)
    env.reset()

    batch_size, num_opes, num_mas = env.ope_ma_adj_batch.shape
    eligible_mask = env.ope_ma_adj_batch > 0  # shape: (batch_size, num_opes, num_mas)

    first_ope_mask = torch.zeros_like(eligible_mask, dtype=torch.bool)
    for b in range(batch_size):
        first_ops = env.num_ope_biases_batch[b]  # first operation of each job
        first_ope_mask[b, first_ops, :] = True
    ma_idle = torch.ones((batch_size, 1, num_mas), dtype=torch.bool)  # all machines idle
    job_ready = torch.ones((batch_size, num_opes, 1), dtype=torch.bool)  # all jobs ready

    initial_eligible_mask = eligible_mask & first_ope_mask & ma_idle & job_ready

    # Generate action list
    batch_size, num_opes, num_mas = initial_eligible_mask.shape
    all_actions = []

    for b in range(batch_size):
        for job in range(env.num_jobs):
            op = env.num_ope_biases_batch[b, job].item()
            mas = torch.nonzero(initial_eligible_mask[b, op, :], as_tuple=False).squeeze()
            if mas.ndim == 0:
                mas = mas.unsqueeze(0)
            actions = [[op, m.item(), job] for m in mas]

            if max_starting_nodes is not None and len(actions) > max_starting_nodes:
                actions = actions[:max_starting_nodes]

            all_actions.extend(actions)

    if len(all_actions) == 0:
        return torch.empty(3, 0, dtype=torch.long), env

    actions_tensor = torch.tensor(all_actions, dtype=torch.long).T  # [3, num_starting_nodes]

    if samples is not None:
        # Repeat each starting node `samples` times
        actions_tensor = actions_tensor.repeat(1, samples)  # [3, num_starting_nodes * samples]
        # Repeat instances to match actions
        env_test_paras["batch_size"] = actions_tensor.shape[1]
        env = gym.make('fjsp-v0', case=instance, env_paras=env_test_paras, data_source='public',
                       is_public_jssp=is_public_jssp)
        env.reset()

    else:
        # For greedy/POMO, repeat instances to match number of starting nodes
        env_test_paras["batch_size"] = actions_tensor.shape[1]
        env = gym.make('fjsp-v0', case=instance, env_paras=env_test_paras, data_source='public',
                       is_public_jssp=is_public_jssp)
        env.reset()

    return actions_tensor, env

def repeat_instances_by_starting_nodes(instances, env_test_paras, is_public_jssp, max_starting_nodes):
    env_test_paras_copy = copy.deepcopy(env_test_paras)
    env_test_paras_copy["batch_size"] = len(instances)
    env = gym.make(
        'fjsp-v0',
        case=instances,  # pass all instances at once
        env_paras=env_test_paras_copy,
        data_source='file'
    )
    env.reset()

    batch_size, num_ops, num_mas = env.ope_ma_adj_batch.shape

    # 2️⃣ Compute eligible starting nodes
    eligible_mask = env.ope_ma_adj_batch > 0  # [batch_size, num_ops, num_mas]

    # First operation mask
    first_ope_mask = torch.zeros_like(eligible_mask, dtype=torch.bool)
    num_ope_biases_tensor = torch.stack([torch.tensor(b) for b in env.num_ope_biases_batch])  # [batch_size, num_jobs]
    # batch indices
    b_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, num_ope_biases_tensor.shape[1])
    # set first operation mask
    first_ope_mask[b_idx, num_ope_biases_tensor] = True

    ma_idle = torch.ones_like(eligible_mask, dtype=torch.bool)
    job_ready = torch.ones_like(eligible_mask, dtype=torch.bool)

    initial_eligible_mask = eligible_mask & first_ope_mask & ma_idle & job_ready

    all_actions = []
    repeated_instances = []
    instance_splits = []

    # 3️⃣ Generate actions per instance in batch
    for b in range(batch_size):
        ops, mas = torch.nonzero(initial_eligible_mask[b], as_tuple=True)
        jobs = torch.zeros_like(ops)
        for j, op in enumerate(env.num_ope_biases_batch[b]):
            mask = ops == op
            jobs[mask] = j

        actions_tensor_instance = torch.stack([ops, mas, jobs], dim=0)

        # Limit number of starting nodes per instance
        if max_starting_nodes is not None and actions_tensor_instance.shape[1] > max_starting_nodes:
            actions_tensor_instance = actions_tensor_instance[:, :max_starting_nodes]
            num_nodes = min(max_starting_nodes, actions_tensor_instance.shape[1])
        else:
            num_nodes = actions_tensor_instance.shape[1]
        if num_nodes == 0:
            continue

        all_actions.append(actions_tensor_instance)
        repeated_instances.extend([instances[b]] * num_nodes)
        instance_splits.append(num_nodes)

    if len(all_actions) == 0:
        # No starting nodes found for any instance
        actions_tensor = torch.empty(3, 0, dtype=torch.long)
    else:
        actions_tensor = torch.cat(all_actions, dim=1)  # [3, total_num_starting_nodes]

    # 4️⃣ Create a new batched environment for repeated instances
    total_batch_size = actions_tensor.shape[1]
    env_test_paras_copy["batch_size"] = total_batch_size
    env = gym.make(
        'fjsp-v0',
        case=repeated_instances,
        env_paras=env_test_paras_copy,
        data_source='file'
    )
    env.reset()

    return repeated_instances, actions_tensor, env, instance_splits
