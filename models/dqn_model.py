import copy
import math
import torch
import torch.nn as nn
import random
from operator import itemgetter
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from network.hgnn import GATedge, MLPsim
from network.mlp import MLPActor, ValueStream
from network.noisy_layer import NoisyLinear
from utils.c51_utils import compute_c51_loss
from utils.iqn_utils import compute_iqn_loss


class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout, noisy=False):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout, noisy))

        if noisy:
            self.project = nn.Sequential(
                nn.ELU(),
                NoisyLinear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
                nn.ELU(),
                NoisyLinear(self.hidden_size_ope, self.hidden_size_ope),
                nn.ELU(),
                NoisyLinear(self.hidden_size_ope, self.out_size_ope),
            )
        else:
            self.project = nn.Sequential(
                nn.ELU(),
                nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
                nn.ELU(),
                nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
                nn.ELU(),
                nn.Linear(self.hidden_size_ope, self.out_size_ope),
            )

    def forward(self, ope_ma_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, batch_idxes, feats):
        '''
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        h = (feats[1], feats[0], feats[0], feats[0])
        # Identity matrix for self-loop of nodes
        self_adj = torch.eye(feats[0].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch[batch_idxes])

        # Calculate an return operation embedding
        adj = (ope_ma_adj_batch[batch_idxes], ope_pre_adj_batch[batch_idxes],
               ope_sub_adj_batch[batch_idxes], self_adj)
        MLP_embeddings = []
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime

    def reset_noise(self):
        for mlp in self.gnn_layers:
            mlp.reset_noise()
        for noisy in self.project:
            if isinstance(noisy, NoisyLinear):
                noisy.reset_noise()


class HGNNScheduler(nn.Module):
    def __init__(self, model_paras, extension_paras, topk=1):
        super(HGNNScheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        self.use_ddqn = extension_paras["use_ddqn"]
        self.use_per = extension_paras["use_per"]
        self.use_dueling = extension_paras["use_dueling"]
        self.use_noisy = extension_paras["use_noisy"]
        self.use_distributional = extension_paras["use_distributional"]
        self.use_n_step = extension_paras["use_n_step"]
        self.use_iqn = extension_paras["use_iqn"]
        self.use_munch = extension_paras["use_munchausen"]

        self.atom_size = extension_paras["distributional_atom_size"] if self.use_distributional else 1
        self.v_min = extension_paras["distributional_v_min"]
        self.v_max = extension_paras["distributional_v_max"]
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)
        self.quant_emb_dim = extension_paras["iqn_quant_emb_dim"]
        self.munch_tau = extension_paras["munch_tau"]
        self.munch_alpha = extension_paras["munch_alpha"]
        self.munch_clip = extension_paras["munch_clip"]
        self.topk = topk
        self.transition = list()

        # Machine node embedding
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                    self.dropout, self.dropout, activation=F.elu, noisy=self.use_noisy))
        for i in range(1,len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                    self.dropout, self.dropout, activation=F.elu, noisy=self.use_noisy))

        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout, self.use_noisy))
        for i in range(len(self.num_heads)-1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout, self.use_noisy))

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim * self.atom_size, self.use_noisy).to(self.device)

        if self.use_dueling:
            self.value_stream = ValueStream(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, 1 * self.atom_size, self.use_noisy)

        if self.use_iqn:
            self.cos_embedding = nn.Linear(extension_paras["iqn_quant_emb_dim"], self.actor_dim)
            self.pis = torch.FloatTensor([np.pi * i for i in range(self.quant_emb_dim)]).view(1, 1, self.quant_emb_dim).to(
                self.device)

    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''
    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if not flag_sample and not flag_train:
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                proc_idxes = torch.nonzero(proc_time[i])
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]
                proc_norm = self.feature_normalize(proc_values)
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        # DRL-S and scheduling dgnn_layersuring training have a consistent number of operations
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            proc_time_norm = self.feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                proc_time_norm)

    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False, num_quantiles=32):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        if flag_train:
            batch_idxes = torch.tensor([0])
        else:
            batch_idxes = state.batch_idxes

        # Raw feats
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_time = state.proc_times_batch[batch_idxes]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_proc = (copy.deepcopy(features[2]))

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            # First Stage, machine node embedding
            # shape: [len(batch_idxes), num_mas, out_size_ma]
            h_mas = self.get_machines[i](state.ope_ma_adj_batch, state.batch_idxes, features)
            features = (features[0], h_mas, features[2])
            # Second Stage, operation node embedding
            # shape: [len(batch_idxes), max(num_opes), out_size_ope]
            h_opes = self.get_operations[i](state.ope_ma_adj_batch, state.ope_pre_adj_batch, state.ope_sub_adj_batch,
                                            state.batch_idxes, features)
            features = (h_opes, features[1], features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ma]
        # There may be different operations for each instance, which cannot be pooled directly by the matrix
        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)  # shape: [len(batch_idxes), d]
        else:
            h_opes_pooled = h_opes.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ope]

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]
        h_jobs = h_opes.gather(1, jobs_gather)
        # Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
        # Matrix indicating whether machine is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        # Matrix indicating whether job is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        # Input of actor MLP
        # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma*2+out_size_ope*2]
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        # h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # deprecated
        mask = eligible.transpose(1, 2).flatten(1)

        if self.use_dueling:
            if self.use_iqn:
                num_jobs = h_actions.size()[2]
                num_mas = h_actions.size()[1]

                batch_size = len(batch_idxes)
                quantiles = torch.rand(batch_size, num_mas * num_jobs, num_quantiles).to(self.device).unsqueeze(-1)
                cos = torch.cos(quantiles * self.pis)
                cos_x = torch.relu(self.cos_embedding(cos))
                h_actions = h_actions.reshape(batch_size, num_mas * num_jobs, 1, self.actor_dim)

                x = (h_actions * cos_x).reshape(num_quantiles, num_mas, num_jobs, self.actor_dim)

                value = self.value_stream(x).squeeze(2, 3)
                advantage = self.actor(x).flatten(1)

                # Get priority index and probability of actions with masking the ineligible actions
                scores = value + advantage - advantage.mean(dim=-1, keepdim=True)

                scores = scores.reshape(batch_size, num_quantiles, -1)
                scores = scores.mean(dim=1)
                # scores[~mask] = float('-inf')
                scores[~mask] = -1e9
                # action_probs = F.softmax(scores, dim=1)
            else:
                if self.use_distributional:
                    value = self.value_stream(h_actions).reshape(-1, 1)
                    advantage = self.actor(h_actions).flatten(1)

                    value = value.view(-1, 1, self.atom_size)
                    advantage = advantage.view(value.size()[0], -1, self.atom_size)

                    q = value + advantage - advantage.mean(dim=-1, keepdim=True)
                    q = F.softmax(q, dim=-1)
                    q = q.clamp(min=1e-3)

                    scores = torch.sum(q * self.support, dim=2)
                    scores[~mask] = -1e9
                else:
                    value = self.value_stream(h_actions).reshape(-1, 1)
                    advantage = self.actor(h_actions).flatten(1)

                    # Get priority index and probability of actions with masking the ineligible actions
                    scores = value + advantage - advantage.mean(dim=-1, keepdim=True)
                    # scores[~mask] = float('-inf')
                    scores[~mask] = -1e9
                    # action_probs = F.softmax(scores, dim=1)
        else:
            if self.use_iqn:
                num_jobs = h_actions.size()[2]
                num_mas = h_actions.size()[1]

                batch_size = len(batch_idxes)
                quantiles = torch.rand(batch_size, num_quantiles).to(self.device).unsqueeze(-1)
                # quantiles = torch.rand(batch_size, num_mas * num_jobs, num_quantiles).to(self.device).unsqueeze(-1)
                cos = torch.cos(quantiles * self.pis)
                cos_x = torch.relu(self.cos_embedding(cos))
                quantile_embedding = cos_x.unsqueeze(1).unsqueeze(1)
                state_embeddings = h_actions.unsqueeze(3)
                fused_embeddings = state_embeddings * quantile_embedding
                # h_actions = h_actions.reshape(batch_size, num_mas * num_jobs, 1, self.actor_dim)

                # x = (h_actions * cos_x).reshape(batch_size, num_quantiles, num_mas, num_jobs, self.actor_dim)

                scores = self.actor(fused_embeddings).reshape(batch_size, num_mas * num_jobs, -1)
                # scores = self.actor(x).flatten(1)
                # scores = scores.reshape(batch_size, num_quantiles, -1)
                scores = scores.mean(dim=2)
                # scores[~mask] = float('-inf')
                scores[~mask] = -1e9
                # action_probs = F.softmax(scores, dim=1)
            else:
                if self.use_distributional:
                    q_atoms = self.actor(h_actions).flatten(1).view(len(batch_idxes), -1, self.atom_size)
                    q = F.softmax(q_atoms, dim=-1)
                    q = q.clamp(min=1e-3)

                    scores = torch.sum(q * self.support, dim=2)
                    scores[~mask] = -1e9
                else:
                    # Get priority index and probability of actions with masking the ineligible actions
                    scores = self.actor(h_actions).flatten(1)
                    # scores[~mask] = float('-inf')
                    scores[~mask] = -1e9
                    # action_probs = F.softmax(scores, dim=1)

        self.transition = ([
            copy.deepcopy(state.ope_ma_adj_batch),
            copy.deepcopy(state.ope_pre_adj_batch),
            copy.deepcopy(state.ope_sub_adj_batch),
            copy.deepcopy(norm_opes),
            copy.deepcopy(norm_mas),
            copy.deepcopy(norm_proc),
            copy.deepcopy(jobs_gather)
        ],)

        return scores, ope_step_batch

    def act(self, state, memories, dones, epsilon, flag_sample=False, flag_train=True, num_quantiles=32):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        scores, ope_step_batch = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train, num_quantiles=num_quantiles)

        if flag_train:
            if self.use_noisy:
                action_indexes = scores.argmax(dim=1)
            else:
                if epsilon > np.random.random():
                    valid_actions = scores != -1e9  # Create a mask for valid actions
                    indices = torch.where(valid_actions[0] == True)
                    random_index = torch.randint(0, len(indices[0]), (1,)).item()
                    action_indexes = torch.tensor([indices[0][random_index].item()])
                else:
                    action_indexes = scores.argmax(dim=1)
        else:
            if flag_sample:
                topk_scores, topk_indices = torch.topk(scores, k=self.topk, dim=1)
                topk_probs = F.softmax(topk_scores, dim=1)
                dist = Categorical(topk_probs)
                topk_sampled_idx = dist.sample().unsqueeze(1)  # Shape: (batch_size, 1)
                action_indexes = topk_indices.gather(1, topk_sampled_idx).squeeze(1)
            # DRL-G, greedily picking actions with the maximum q_value
            else:
                action_indexes = scores.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]

        # Store data in memory during training
        if flag_train == True:
            self.transition += (action_indexes,)

        return torch.stack((opes, mas, jobs), dim=1).t()

    def add_next_state(self, state, memories, flag_sample=False, flag_train=True):
        # Uncompleted instances
        batch_idxes = torch.tensor([0])
        # Raw feats
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_time = state.proc_times_batch[batch_idxes]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_proc = (copy.deepcopy(features[2]))

        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)

        self.transition += ([
            copy.deepcopy(state.ope_ma_adj_batch),
            copy.deepcopy(state.ope_pre_adj_batch),
            copy.deepcopy(state.ope_sub_adj_batch),
            copy.deepcopy(norm_opes),
            copy.deepcopy(norm_mas),
            copy.deepcopy(norm_proc),
            copy.deepcopy(ope_step_batch),
            copy.deepcopy(state.mask_ma_procing_batch),
            copy.deepcopy(state.mask_job_procing_batch),
            copy.deepcopy(state.mask_job_finish_batch),
        ],)

class Model:
    def __init__(self, model_paras, train_paras, extension_paras, topk):
        self.lr = train_paras["lr"]  # learning rate
        self.gamma = train_paras["gamma"]  # discount factor
        self.device = model_paras["device"]  # PyTorch device

        self.use_ddqn = extension_paras["use_ddqn"]
        self.use_per = extension_paras["use_per"]
        self.use_dueling = extension_paras["use_dueling"]
        self.use_noisy = extension_paras["use_noisy"]
        self.use_distributional = extension_paras["use_distributional"]
        self.use_n_step = extension_paras["use_n_step"]
        self.use_iqn = extension_paras["use_iqn"]
        self.use_munch = extension_paras["use_munchausen"]

        self.v_max = extension_paras["distributional_v_max"]
        self.v_min = extension_paras["distributional_v_min"]
        self.atom_size = extension_paras["distributional_atom_size"]
        self.batch_size = train_paras["minibatch_size"]
        self.n_step_horizon = extension_paras["n_step_horizon"]
        self.num_tau_samples = extension_paras["iqn_num_tau_samples"]
        self.num_quant_samples = extension_paras["iqn_num_quant_samples"]
        self.num_tau_prime_samples = extension_paras["iqn_num_tau_prime_samples"]
        self.munch_tau = extension_paras["munch_tau"]
        self.munch_alpha = extension_paras["munch_alpha"]
        self.munch_clip = extension_paras["munch_clip"]

        self.online_network = HGNNScheduler(model_paras, extension_paras, topk).to(self.device)
        self.target_network = copy.deepcopy(self.online_network)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)

    def update_model(self, memory, memory_n=None):
        samples = memory.sample_batch()
        idxs = samples["idxs"]

        if self.use_per:
            weights = torch.FloatTensor(
                samples["weights"].reshape(-1, 1)
            ).to(self.device)

            elementwise_loss = self.compute_loss(samples, self.gamma)
            loss = torch.mean(elementwise_loss * weights)

            if self.use_n_step:
                samples = memory_n.sample_batch_from_idxs(idxs)
                elementwise_loss_n_loss = self.compute_loss(samples, self.gamma ** self.n_step_horizon)
                elementwise_loss += elementwise_loss_n_loss

                loss = torch.mean(elementwise_loss * weights)

            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            priorities = loss_for_prior + 1e-6
            new_priorities = priorities[0] if len(priorities) == 1 else priorities
            memory.update_priorities(idxs, new_priorities)
        else:
            loss = self.compute_loss(samples, self.gamma)

            if self.use_n_step:
                samples = memory_n.sample_batch_from_idxs(idxs)
                n_loss = self.compute_loss(samples, self.gamma ** self.n_step_horizon)
                loss += n_loss

        self.optimizer.zero_grad()
        loss.backward()

        if self.use_dueling:
            clip_grad_norm_(self.online_network.parameters(), 10.0)

        self.optimizer.step()

        if self.use_noisy:
            for layer in self.online_network.get_machines:
                layer.reset_noise()
            for layer in self.online_network.get_operations:
                layer.reset_noise()
            self.online_network.actor.reset_noise()

            for layer in self.target_network.get_machines:
                layer.reset_noise()
            for layer in self.target_network.get_operations:
                layer.reset_noise()
            self.target_network.actor.reset_noise()

            if self.use_dueling:
                self.online_network.value_stream.reset_noise()
                self.target_network.value_stream.reset_noise()

    def compute_loss(self, samples, discount):
        if self.use_iqn:
            loss = compute_iqn_loss(self.online_network, self.target_network, self.num_tau_samples,
                                    self.num_quant_samples, self.num_tau_prime_samples, samples, discount)
        elif self.use_distributional:
            loss = compute_c51_loss(self.v_max, self.v_min, self.atom_size, samples, self.online_network,
                                    self.target_network, self.batch_size, discount, self.device)
        else:
            ope_ma_adj = torch.stack(samples["ope_ma_adj"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            ope_pre_adj = torch.stack(samples["ope_pre_adj"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            ope_sub_adj = torch.stack(samples["ope_sub_adj"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            raw_opes = torch.stack(samples["raw_opes"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            raw_mas = torch.stack(samples["raw_mas"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            proc_time = torch.stack(samples["proc_time"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            jobs_gather = torch.stack(samples["jobs_gather"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))

            rewards = torch.stack(samples["rewards"], dim=0).to(torch.device("cuda:0"))
            dones = torch.stack(samples["dones"], dim=0).to(torch.device("cuda:0"))
            actions = torch.stack(samples["actions"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))

            next_ope_ma_adj = torch.stack(samples["next_ope_ma_adj"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            next_ope_pre_adj = torch.stack(samples["next_ope_pre_adj"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            next_ope_sub_adj = torch.stack(samples["next_ope_sub_adj"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            next_raw_opes = torch.stack(samples["next_raw_opes"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            next_raw_mas = torch.stack(samples["next_raw_mas"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            next_proc_time = torch.stack(samples["next_proc_time"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            next_ope_step_batch = torch.stack(samples["next_ope_step_batch"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            next_mask_mas = torch.stack(samples["next_mask_mas"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            next_mask_job_procing = torch.stack(samples["next_mask_job_procing"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))
            next_mask_job_finish = torch.stack(samples["next_mask_job_finish"], dim=0).transpose(0, 1).flatten(0, 1).to(torch.device("cuda:0"))

            curr_q_value = self.compute_curr_q_value(ope_ma_adj, raw_opes, raw_mas, proc_time, ope_pre_adj, ope_sub_adj,
                                                     jobs_gather, self.online_network)
            curr_q_value = curr_q_value.gather(1, actions.unsqueeze(1))

            if self.use_munch:
                next_q_values = self.compute_next_q_value(next_ope_ma_adj, next_ope_pre_adj, next_ope_sub_adj, next_raw_opes,
                                                     next_raw_mas, next_proc_time, next_ope_step_batch, next_mask_mas,
                                                     next_mask_job_procing, next_mask_job_finish)

                y = next_q_values - next_q_values.max(dim=1, keepdim=True)[0]
                current_q_values = self.compute_curr_q_value(ope_ma_adj, raw_opes, raw_mas, proc_time, ope_pre_adj, ope_sub_adj,
                                                     jobs_gather, self.target_network)
                y_curr = current_q_values - current_q_values.max(dim=1, keepdim=True)[0]

                tau_log_pi_next = next_q_values - next_q_values.max(dim=1, keepdim=True)[0] + self.munch_tau * torch.log(
                    torch.sum(torch.exp(y / self.munch_tau), dim=1, keepdim=True))
                replay_log_policy = current_q_values - current_q_values.max(dim=1, keepdim=True)[
                    0] + self.munch_tau * torch.log(torch.sum(torch.exp(y_curr / self.munch_tau), dim=1, keepdim=True))
                pi_target = F.softmax(y / self.munch_tau, dim=1)

                replay_next_qt_softmax = torch.sum(
                    (next_q_values - tau_log_pi_next) * pi_target, dim=1, keepdim=True
                )

                tau_log_pi_a = torch.gather(replay_log_policy, dim=1, index=actions.unsqueeze(-1))
                tau_log_pi_a = torch.clip(tau_log_pi_a, min=self.munch_clip, max=1)

                mask = ~dones
                munchausen_term = self.munch_alpha * tau_log_pi_a

                target = (rewards + munchausen_term + mask.view(-1, 1) * discount * replay_next_qt_softmax)
            else:
                next_q_value = self.compute_next_q_value(next_ope_ma_adj, next_ope_pre_adj, next_ope_sub_adj, next_raw_opes,
                                                         next_raw_mas, next_proc_time, next_ope_step_batch, next_mask_mas,
                                                         next_mask_job_procing, next_mask_job_finish)

                mask = ~dones
                target = (rewards + discount * next_q_value * mask).to(self.device)

            if self.use_per:
                loss = F.mse_loss(curr_q_value, target, reduction="none")
            else:
                loss = F.mse_loss(curr_q_value, target)

        return loss

    def compute_next_scores(self, next_ope_ma_adj, next_ope_pre_adj, next_ope_sub_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_step_batch, next_mask_mas,
                                                 next_mask_job_procing, next_mask_job_finish, target=True):
        model = self.target_network if target else self.online_network

        batch_idxes = torch.arange(0, next_ope_ma_adj.size(-3)).long()

        features = (next_raw_opes, next_raw_mas, next_proc_time)

        # L iterations of the HGNN
        for i in range(len(self.target_network.num_heads)):
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
                                                                                                           -1))[batch_idxes])
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
        # h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # deprecated
        mask = eligible.transpose(1, 2).flatten(1)

        if self.use_dueling:
            value = model.value_stream(h_actions).reshape(-1, 1)
            advantage = model.actor(h_actions).flatten(1)

            scores = value + advantage - advantage.mean(dim=-1, keepdim=True)
            scores[~mask] = -1e9
        else:
            # Get priority index and probability of actions with masking the ineligible actions
            scores = model.actor(h_actions).flatten(1)
            scores[~mask] = -1e9

        return scores

    def compute_next_q_value(self, next_ope_ma_adj, next_ope_pre_adj, next_ope_sub_adj, next_raw_opes, next_raw_mas, next_proc_time, next_ope_step_batch, next_mask_mas,
                                                 next_mask_job_procing, next_mask_job_finish):

        scores = self.compute_next_scores(next_ope_ma_adj, next_ope_pre_adj, next_ope_sub_adj, next_raw_opes,
                                          next_raw_mas, next_proc_time, next_ope_step_batch, next_mask_mas,
                                          next_mask_job_procing, next_mask_job_finish)

        if self.use_munch:
            next_q_value = scores
        else:
            if not self.use_ddqn:
                next_q_value = scores.max(dim=1, keepdim=True)[0].detach()
            else:
                online_scores = self.compute_next_scores(next_ope_ma_adj, next_ope_pre_adj, next_ope_sub_adj, next_raw_opes,
                                              next_raw_mas, next_proc_time, next_ope_step_batch, next_mask_mas,
                                              next_mask_job_procing, next_mask_job_finish, target=False)

                next_q_value = scores.gather(1, online_scores.argmax(dim=1, keepdim=True)).detach()

        return next_q_value

    def compute_curr_q_value(self, ope_ma_adj, raw_opes, raw_mas, proc_time, ope_pre_adj, ope_sub_adj, jobs_gather, model):
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
        # h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)

        if self.use_dueling:
            value = model.value_stream(h_actions).reshape(-1, 1)
            advantage = model.actor(h_actions).flatten(1)

            scores = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            scores = model.actor(h_actions).flatten(1)
        curr_q_value = scores

        return curr_q_value
