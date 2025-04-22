import torch
import torch.nn.functional as F

def compute_iqn_next_scores(model, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                            next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish, num_quantiles):
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
        num_jobs = h_actions.size()[2]
        num_mas = h_actions.size()[1]

        batch_size = len(batch_idxes)
        quantiles = torch.rand(batch_size, num_mas * num_jobs, num_quantiles).to(model.device).unsqueeze(-1)
        cos = torch.cos(quantiles * model.pis)
        cos_x = torch.relu(model.cos_embedding(cos))
        h_actions = h_actions.reshape(batch_size, num_mas * num_jobs, 1, model.actor_dim)

        x = (h_actions * cos_x).reshape(num_quantiles, num_mas, num_jobs, model.actor_dim)

        value = model.value_stream(x).squeeze(2, 3)
        advantage = model.actor(x).flatten(1)

        # Get priority index and probability of actions with masking the ineligible actions
        scores = value + advantage - advantage.mean(dim=-1, keepdim=True)

        scores = scores.reshape(batch_size, num_quantiles, -1)
        scores = scores
        # scores[~mask] = float('-inf')
        # scores[~mask] = -1e9
        # action_probs = F.softmax(scores, dim=1)
    else:
        num_jobs = h_actions.size()[2]
        num_mas = h_actions.size()[1]

        batch_size = len(batch_idxes)
        quantiles = torch.rand(batch_size, num_quantiles).to(model.device).unsqueeze(-1)
        # quantiles = torch.rand(batch_size, num_mas * num_jobs, num_quantiles).to(self.device).unsqueeze(-1)
        cos = torch.cos(quantiles * model.pis)
        cos_x = torch.relu(model.cos_embedding(cos))
        quantile_embedding = cos_x.unsqueeze(1).unsqueeze(1)
        state_embeddings = h_actions.unsqueeze(3)
        fused_embeddings = state_embeddings * quantile_embedding
        # h_actions = h_actions.reshape(batch_size, num_mas * num_jobs, 1, model.actor_dim)
        # x = (h_actions * cos_x).reshape(batch_size, num_quantiles, num_mas, num_jobs, model.actor_dim)

        scores = model.actor(fused_embeddings).reshape(batch_size, num_mas * num_jobs, -1)
        # scores = model.actor(x).flatten(1)
        # scores = scores.reshape(batch_size, num_quantiles, -1)
        # scores = scores
        # scores[~mask] = float('-inf')
        # scores[~mask] = -1e9
        # action_probs = F.softmax(scores, dim=1)

    return scores


def compute_iqn_scores(model, ope_ma_adj, raw_opes, raw_mas, proc_time, ope_pre_adj, ope_sub_adj, jobs_gather, num_quantiles):
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
        num_jobs = h_actions.size()[2]
        num_mas = h_actions.size()[1]

        batch_size = len(batch_idxes)
        quantiles = torch.rand(batch_size, num_mas * num_jobs, num_quantiles).to(model.device).unsqueeze(-1)
        cos = torch.cos(quantiles * model.pis)
        cos_x = torch.relu(model.cos_embedding(cos))
        h_actions = h_actions.reshape(batch_size, num_mas * num_jobs, 1, model.actor_dim)

        x = (h_actions * cos_x).reshape(num_quantiles, num_mas, num_jobs, model.actor_dim)

        value = model.value_stream(x).squeeze(2, 3)
        advantage = model.actor(x).flatten(1)

        # Get priority index and probability of actions with masking the ineligible actions
        scores = value + advantage - advantage.mean(dim=-1, keepdim=True)

        scores = scores.reshape(batch_size, num_quantiles, -1)
        scores = scores
        # scores[~mask] = float('-inf')
        # scores[~mask] = -1e9
        # action_probs = F.softmax(scores, dim=1)
    else:
        num_jobs = h_actions.size()[2]
        num_mas = h_actions.size()[1]

        batch_size = len(batch_idxes)
        quantiles = torch.rand(batch_size, num_quantiles).to(model.device).unsqueeze(-1)
        # quantiles = torch.rand(batch_size, num_mas * num_jobs, num_quantiles).to(self.device).unsqueeze(-1)
        cos = torch.cos(quantiles * model.pis)
        cos_x = torch.relu(model.cos_embedding(cos))
        quantile_embedding = cos_x.unsqueeze(1).unsqueeze(1)
        state_embeddings = h_actions.unsqueeze(3)
        fused_embeddings = state_embeddings * quantile_embedding
        # h_actions = h_actions.reshape(batch_size, num_mas * num_jobs, 1, model.actor_dim)
        # x = (h_actions * cos_x).reshape(batch_size, num_quantiles, num_mas, num_jobs, model.actor_dim)

        scores = model.actor(fused_embeddings).reshape(batch_size, num_mas * num_jobs, -1)
        # scores = model.actor(x).flatten(1)
        # scores = scores.reshape(batch_size, num_quantiles, -1)
        # scores = scores
        # scores[~mask] = float('-inf')
        # scores[~mask] = -1e9
        # action_probs = F.softmax(scores, dim=1)

    return scores

def compute_iqn_loss(online_network, target_network, num_tau_samples,
                     num_quant_samples, num_tau_prime_samples, samples, gamma):
    ope_ma_adj = torch.stack(samples["ope_ma_adj"], dim=0).transpose(0, 1).flatten(0, 1)
    ope_pre_adj = torch.stack(samples["ope_pre_adj"], dim=0).transpose(0, 1).flatten(0, 1)
    ope_sub_adj = torch.stack(samples["ope_sub_adj"], dim=0).transpose(0, 1).flatten(0, 1)
    raw_opes = torch.stack(samples["raw_opes"], dim=0).transpose(0, 1).flatten(0, 1)
    raw_mas = torch.stack(samples["raw_mas"], dim=0).transpose(0, 1).flatten(0, 1)
    proc_time = torch.stack(samples["proc_time"], dim=0).transpose(0, 1).flatten(0, 1)
    jobs_gather = torch.stack(samples["jobs_gather"], dim=0).transpose(0, 1).flatten(0, 1)

    rewards = torch.stack(samples["rewards"], dim=0)
    dones = torch.stack(samples["dones"], dim=0)
    actions = torch.stack(samples["actions"], dim=0)

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

    batch_size = len(torch.arange(0, ope_ma_adj.size(-3)).long())

    if online_network.use_munch:
        replay_net_target_quantile_values = compute_iqn_next_scores(target_network, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                            next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish, num_tau_prime_samples).mean(dim=1)

        target_next_quantile_values_action = compute_iqn_next_scores(target_network, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                            next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish, num_quant_samples)
        _replay_next_target_q_values = target_next_quantile_values_action.mean(dim=1)

        q_state_values = compute_iqn_scores(target_network, ope_ma_adj, raw_opes, raw_mas, proc_time,
                                                      ope_pre_adj, ope_sub_adj, jobs_gather, num_quant_samples).mean(dim=1)
        _replay_target_q_values = q_state_values

        y = _replay_next_target_q_values - _replay_next_target_q_values.max(dim=0, keepdim=True)[0]
        y_curr = _replay_target_q_values - _replay_target_q_values.max(dim=0, keepdim=True)[0]
        replay_next_log_policy = _replay_next_target_q_values - _replay_next_target_q_values.max(dim=0, keepdim=True)[
            0] + online_network.munch_alpha * torch.log(
            torch.sum(torch.exp(y / online_network.munch_tau), dim=0, keepdim=True))
        replay_next_policy = F.softmax(y / online_network.munch_tau, dim=0)
        replay_log_policy = _replay_target_q_values - _replay_target_q_values.max(dim=0, keepdim=True)[
            0] + online_network.munch_tau * torch.log(
            torch.sum(torch.exp(y_curr / online_network.munch_tau), dim=0, keepdim=True))

        tau_log_pi_a = torch.gather(replay_log_policy, dim=0, index=actions)
        tau_log_pi_a = torch.clip(tau_log_pi_a, min=online_network.munch_clip, max=0)
        munchausen_term = online_network.munch_alpha * tau_log_pi_a

        rewards += munchausen_term
        weighted_logits = replay_next_policy * (replay_net_target_quantile_values - replay_next_log_policy)

        target_quantile_vals = weighted_logits.sum(dim=1)

        mask = ~dones
        target_quantile_vals = rewards.unsqueeze(-1) + (
                gamma * target_quantile_vals * mask.unsqueeze(-1))
    else:
        if not online_network.use_ddqn:
            target_quantile_vals = compute_iqn_next_scores(target_network, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                                next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish, num_tau_prime_samples)
            outputs_action = compute_iqn_next_scores(target_network, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                                next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish, num_quant_samples)
            outputs_action = outputs_action.detach().max(1)[1].unsqueeze(1)  # (batch_size, 1, N)
            target_quantile_vals = target_quantile_vals.gather(1, outputs_action)
        else:
            target_quantile_vals = compute_iqn_next_scores(target_network, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                                next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish, num_tau_prime_samples)
            outputs_action = compute_iqn_next_scores(online_network, next_ope_ma_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_pre_adj,
                                next_ope_sub_adj, next_ope_step_batch, next_mask_mas, next_mask_job_procing, next_mask_job_finish, num_quant_samples)
            outputs_action = outputs_action.detach().max(1)[1].unsqueeze(1)  # (batch_size, 1, N)
            target_quantile_vals = target_quantile_vals.gather(1, outputs_action)

        mask = ~dones
        target_quantile_vals = rewards.unsqueeze(-1) + (
                gamma * target_quantile_vals * mask.unsqueeze(-1))

    chosen_action_quantile_values = compute_iqn_scores(online_network, ope_ma_adj, raw_opes, raw_mas, proc_time,
                                                      ope_pre_adj, ope_sub_adj, jobs_gather, num_tau_samples)
    chosen_action_quantile_values = chosen_action_quantile_values.gather(1, actions.unsqueeze(-1).expand(batch_size,
                                                                                                        num_tau_samples,
                                                                                                        1))

    quantiles = torch.rand(batch_size, num_tau_samples).to(online_network.device).unsqueeze(-1)
    bellman_errors = target_quantile_vals - chosen_action_quantile_values
    huber_loss = abs(bellman_errors.abs() <= 1) * 0.5 * bellman_errors.pow(2) + abs(
        bellman_errors.abs() > 1) * 1 * (bellman_errors.abs() - 0.5 * 1)
    quantil_l = abs(quantiles - (bellman_errors.detach() < 0).float()) * huber_loss / 1

    loss = quantil_l.sum(dim=1).mean(dim=1)
    if not online_network.use_per:
        loss = loss.mean()

    return loss
