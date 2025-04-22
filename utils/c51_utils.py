import torch
import torch.nn.functional as F

def compute_c51_next_scores(model, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                            next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish):

    batch_idxes = torch.arange(0, next_ope_ma_adj.size(-3)).long()

    features = (next_raw_opes, next_raw_mas, next_proc_time)

    # L iterations of the HGNN
    for i in range(len(model.num_heads)):
        h_mas = model.get_machines[i](next_ope_ma_adj, batch_idxes, features)
        features = (features[0], h_mas, features[2])
        h_opes = model.get_operations[i](next_ope_ma_adj, next_ope_pre_adj, next_ope_sub_adj, batch_idxes, features)
        features = (h_opes, features[1], features[2])

    # Stacking and pooling
    h_mas_pooled = h_mas.mean(dim=-2)
    h_opes_pooled = h_opes.mean(dim=-2)

    # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
    jobs_gather = next_ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]

    h_jobs = h_opes.gather(1, jobs_gather)
    # Matrix indicating whether processing is possible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    eligible_proc = next_ope_ma_adj[batch_idxes].gather(1, next_ope_step_batch[..., :, None].expand(-1, -1,
                                                                                                    next_ope_ma_adj.size(
                                                                                                        -1))[
        batch_idxes])
    h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, next_proc_time.size(-1), -1)
    h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
    h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
    h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
    # Matrix indicating whether machine is eligible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    ma_eligible = ~next_mask_mas[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
    # Matrix indicating whether job is eligible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    job_eligible = ~(next_mask_job_procing[batch_idxes] +
                     next_mask_job_finish[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
    # shape: [len(batch_idxes), num_jobs, num_mas]
    eligible = job_eligible & ma_eligible & (eligible_proc == 1)
    if (~(eligible)).all():
        print("No eligible O-M pair!")
        return
    # Input of actor MLP
    # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma*2+out_size_ope*2]
    h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                          dim=-1).transpose(1, 2)
    h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # deprecated
    mask = eligible.transpose(1, 2).flatten(1)

    if model.use_dueling:
        value = model.value_stream(h_actions).reshape(-1, 1)
        advantage = model.actor(h_actions).flatten(1)

        value = value.view(-1, 1, model.atom_size)
        advantage = advantage.view(value.size()[0], -1, model.atom_size)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        q = F.softmax(q, dim=-1)
        q = q.clamp(min=1e-3)
    else:
        q_atoms = model.actor(h_actions).flatten(1).view(next_ope_ma_adj.size(-3), -1, model.atom_size)
        q = F.softmax(q_atoms, dim=-1)
        q = q.clamp(min=1e-3)

    return q, mask

def compute_c51_scores(model, ope_ma_adj, raw_opes, raw_mas, proc_time, ope_pre_adj, ope_sub_adj, jobs_gather):
    batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()

    features = (raw_opes, raw_mas, proc_time)

    # L iterations of the HGNN
    for i in range(len(model.num_heads)):
        h_mas = model.get_machines[i](ope_ma_adj, batch_idxes, features)
        features = (features[0], h_mas, features[2])
        h_opes = model.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes, features)
        features = (h_opes, features[1], features[2])

    # Stacking and pooling
    h_mas_pooled = h_mas.mean(dim=-2)
    h_opes_pooled = h_opes.mean(dim=-2)

    # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
    h_jobs = h_opes.gather(1, jobs_gather)
    h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
    h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
    h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
    h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

    h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                          dim=-1).transpose(1, 2)
    h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)

    if model.use_dueling:
        value = model.value_stream(h_actions).reshape(-1, 1)
        advantage = model.actor(h_actions).flatten(1)

        value = value.view(-1, 1, model.atom_size)
        advantage = advantage.view(value.size()[0], -1, model.atom_size)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        q = F.softmax(q, dim=-1)
        q = q.clamp(min=1e-3)
    else:
        q_atoms = model.actor(h_actions).flatten(1).view(ope_ma_adj.size(-3), -1, model.atom_size)
        q = F.softmax(q_atoms, dim=-1)
        q = q.clamp(min=1e-3)

    return q


def compute_c51_loss(v_max, v_min, atom_size, samples, online_network, target_network, batch_size, gamma, device):
    ope_ma_adj = torch.stack(samples["ope_ma_adj"], dim=0).transpose(0, 1).flatten(0, 1)
    ope_pre_adj = torch.stack(samples["ope_pre_adj"], dim=0).transpose(0, 1).flatten(0, 1)
    ope_sub_adj = torch.stack(samples["ope_sub_adj"], dim=0).transpose(0, 1).flatten(0, 1)
    raw_opes = torch.stack(samples["raw_opes"], dim=0).transpose(0, 1).flatten(0, 1)
    raw_mas = torch.stack(samples["raw_mas"], dim=0).transpose(0, 1).flatten(0, 1)
    proc_time = torch.stack(samples["proc_time"], dim=0).transpose(0, 1).flatten(0, 1)
    jobs_gather = torch.stack(samples["jobs_gather"], dim=0).transpose(0, 1).flatten(0, 1)

    rewards = torch.stack(samples["rewards"], dim=0)
    dones = torch.stack(samples["dones"], dim=0)
    actions = torch.stack(samples["actions"], dim=0).transpose(0, 1).flatten(0, 1)

    next_ope_ma_adj = torch.stack(samples["next_ope_ma_adj"], dim=0).transpose(0, 1).flatten(0, 1)
    next_ope_pre_adj = torch.stack(samples["next_ope_pre_adj"], dim=0).transpose(0, 1).flatten(0, 1)
    next_ope_sub_adj = torch.stack(samples["next_ope_sub_adj"], dim=0).transpose(0, 1).flatten(0, 1)
    next_raw_opes = torch.stack(samples["next_raw_opes"], dim=0).transpose(0, 1).flatten(0, 1)
    next_raw_mas = torch.stack(samples["next_raw_mas"], dim=0).transpose(0, 1).flatten(0, 1)
    next_proc_time = torch.stack(samples["next_proc_time"], dim=0).transpose(0, 1).flatten(0, 1)
    next_ope_step_batch = torch.stack(samples["next_ope_step_batch"], dim=0).transpose(0, 1).flatten(0, 1)
    next_mask_mas = torch.stack(samples["next_mask_mas"], dim=0).transpose(0, 1).flatten(0, 1)
    next_mask_job_procing = torch.stack(samples["next_mask_job_procing"], dim=0).transpose(0, 1).flatten(0, 1)
    next_mask_job_finish = torch.stack(samples["next_mask_job_finish"], dim=0).transpose(0, 1).flatten(0, 1)

    delta_z = float(v_max - v_min) / (atom_size - 1)

    with torch.no_grad():
        scores_next, next_scores_mask = compute_c51_next_scores(online_network if online_network.use_ddqn else target_network, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                            next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish)
        scores_next = torch.sum(scores_next * online_network.support, dim=2)
        scores_next[~next_scores_mask] = -1e9
        next_action = scores_next.argmax(1)

        next_dist, _ = compute_c51_next_scores(target_network, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                            next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish)
        next_dist = next_dist[range(batch_size), next_action]

        mask = ~dones
        t_z = rewards + mask * gamma * online_network.support
        t_z = t_z.clamp(min=v_min, max=v_max)
        b = (t_z - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = (
            torch.linspace(
                0, (batch_size - 1) * atom_size, batch_size
            ).long()
            .unsqueeze(1)
            .expand(batch_size, atom_size)
            .to(device)
        )

        proj_dist = torch.zeros(next_dist.size(), device=device)
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

    dist = compute_c51_scores(online_network, ope_ma_adj, raw_opes, raw_mas, proc_time, ope_pre_adj, ope_sub_adj, jobs_gather)
    log_p = torch.log(dist[range(batch_size), actions])

    if online_network.use_per == 1:
        loss = -(proj_dist * log_p).sum(1)
    else:
        loss = -(proj_dist * log_p).sum(1).mean()

    return loss
