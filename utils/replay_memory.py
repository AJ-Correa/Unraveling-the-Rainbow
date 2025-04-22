import numpy as np
import random
from collections import deque
from typing import Dict
from operator import itemgetter
from typing import Deque, Dict, List, Tuple
import torch
from utils.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    """A simple deque replay buffer."""

    def __init__(self, max_size=1000000, batch_size=32, gamma=0.95, n_step_horizon=1, use_n_step=False, to_cpu=False):
        self.max_size = max_size
        self.to_cpu = to_cpu

        self.rewards = deque(maxlen=max_size)
        self.is_terminals = deque(maxlen=max_size)
        self.action_indexes = deque(maxlen=max_size)

        self.ope_ma_adj = deque(maxlen=max_size)
        self.ope_pre_adj = deque(maxlen=max_size)
        self.ope_sub_adj = deque(maxlen=max_size)
        self.raw_opes = deque(maxlen=max_size)
        self.raw_mas = deque(maxlen=max_size)
        self.proc_time = deque(maxlen=max_size)
        self.jobs_gather = deque(maxlen=max_size)

        self.next_ope_ma_adj = deque(maxlen=max_size)
        self.next_ope_pre_adj = deque(maxlen=max_size)
        self.next_ope_sub_adj = deque(maxlen=max_size)
        self.next_raw_opes = deque(maxlen=max_size)
        self.next_raw_mas = deque(maxlen=max_size)
        self.next_proc_time = deque(maxlen=max_size)
        self.next_ope_step_batch = deque(maxlen=max_size)
        self.next_mask_ma = deque(maxlen=max_size)
        self.next_mask_job_procing = deque(maxlen=max_size)
        self.next_mask_job_finish = deque(maxlen=max_size)

        self.batch_size = batch_size
        self.use_n_step = use_n_step

        if self.use_n_step:
            self.n_step_buffer = deque(maxlen=n_step_horizon)
            self.n_step_horizon = n_step_horizon
            self.gamma = gamma

    def store(self, transition):
        obs, act, next_obs, rew, done = transition
        if self.use_n_step:
            self.n_step_buffer.append(transition)

            if len(self.n_step_buffer) < self.n_step_horizon:
                return ()

            # make a n-step transition
            rew, next_obs, done = self._get_n_step_info(
                self.n_step_buffer, self.gamma)

            obs, act = self.n_step_buffer[0][:2]
        
        if self.to_cpu:
            self.ope_ma_adj.append(obs[0].to(torch.device("cpu")))
            self.ope_pre_adj.append(obs[1].to(torch.device("cpu")))
            self.ope_sub_adj.append(obs[2].to(torch.device("cpu")))
            self.raw_opes.append(obs[3].to(torch.device("cpu")))
            self.raw_mas.append(obs[4].to(torch.device("cpu")))
            self.proc_time.append(obs[5].to(torch.device("cpu")))
            self.jobs_gather.append(obs[6].to(torch.device("cpu")))

            self.next_ope_ma_adj.append(next_obs[0].to(torch.device("cpu")))
            self.next_ope_pre_adj.append(next_obs[1].to(torch.device("cpu")))
            self.next_ope_sub_adj.append(next_obs[2].to(torch.device("cpu")))
            self.next_raw_opes.append(next_obs[3].to(torch.device("cpu")))
            self.next_raw_mas.append(next_obs[4].to(torch.device("cpu")))
            self.next_proc_time.append(next_obs[5].to(torch.device("cpu")))
            self.next_ope_step_batch.append(next_obs[6].to(torch.device("cpu")))
            self.next_mask_ma.append(next_obs[7].to(torch.device("cpu")))
            self.next_mask_job_procing.append(next_obs[8].to(torch.device("cpu")))
            self.next_mask_job_finish.append(next_obs[9].to(torch.device("cpu")))

            self.action_indexes.append(act.to(torch.device("cpu")))
            self.rewards.append(rew.to(torch.device("cpu")))
            self.is_terminals.append(done.to(torch.device("cpu")))
        else:
            self.ope_ma_adj.append(obs[0])
            self.ope_pre_adj.append(obs[1])
            self.ope_sub_adj.append(obs[2])
            self.raw_opes.append(obs[3])
            self.raw_mas.append(obs[4])
            self.proc_time.append(obs[5])
            self.jobs_gather.append(obs[6])

            self.next_ope_ma_adj.append(next_obs[0])
            self.next_ope_pre_adj.append(next_obs[1])
            self.next_ope_sub_adj.append(next_obs[2])
            self.next_raw_opes.append(next_obs[3])
            self.next_raw_mas.append(next_obs[4])
            self.next_proc_time.append(next_obs[5])
            self.next_ope_step_batch.append(next_obs[6])
            self.next_mask_ma.append(next_obs[7])
            self.next_mask_job_procing.append(next_obs[8])
            self.next_mask_job_finish.append(next_obs[9])

            self.action_indexes.append(act)
            self.rewards.append(rew)
            self.is_terminals.append(done)
            

        if self.use_n_step:
            return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        # Randomly sample a batch of indices from the deque
        idxs = np.random.choice(len(self.ope_ma_adj), size=self.batch_size, replace=False)

        return dict(
            ope_ma_adj=[self.ope_ma_adj[i] for i in idxs],
            ope_pre_adj=[self.ope_pre_adj[i] for i in idxs],
            ope_sub_adj=[self.ope_sub_adj[i] for i in idxs],
            raw_opes=[self.raw_opes[i] for i in idxs],
            raw_mas=[self.raw_mas[i] for i in idxs],
            proc_time=[self.proc_time[i] for i in idxs],
            jobs_gather=[self.jobs_gather[i] for i in idxs],
            rewards=[self.rewards[i] for i in idxs],
            dones=[self.is_terminals[i] for i in idxs],
            actions=[self.action_indexes[i] for i in idxs],
            next_ope_ma_adj=[self.next_ope_ma_adj[i] for i in idxs],
            next_ope_pre_adj=[self.next_ope_pre_adj[i] for i in idxs],
            next_ope_sub_adj=[self.next_ope_sub_adj[i] for i in idxs],
            next_raw_opes=[self.next_raw_opes[i] for i in idxs],
            next_raw_mas=[self.next_raw_mas[i] for i in idxs],
            next_proc_time=[self.next_proc_time[i] for i in idxs],
            next_ope_step_batch=[self.next_ope_step_batch[i] for i in idxs],
            next_mask_mas=[self.next_mask_ma[i] for i in idxs],
            next_mask_job_procing=[self.next_mask_job_procing[i] for i in idxs],
            next_mask_job_finish=[self.next_mask_job_finish[i] for i in idxs],
            idxs=idxs
        )

    def sample_batch_from_idxs(
            self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            ope_ma_adj=[self.ope_ma_adj[i] for i in idxs],
            ope_pre_adj=[self.ope_pre_adj[i] for i in idxs],
            ope_sub_adj=[self.ope_sub_adj[i] for i in idxs],
            raw_opes=[self.raw_opes[i] for i in idxs],
            raw_mas=[self.raw_mas[i] for i in idxs],
            proc_time=[self.proc_time[i] for i in idxs],
            jobs_gather=[self.jobs_gather[i] for i in idxs],
            rewards=[self.rewards[i] for i in idxs],
            dones=[self.is_terminals[i] for i in idxs],
            actions=[self.action_indexes[i] for i in idxs],
            next_ope_ma_adj=[self.next_ope_ma_adj[i] for i in idxs],
            next_ope_pre_adj=[self.next_ope_pre_adj[i] for i in idxs],
            next_ope_sub_adj=[self.next_ope_sub_adj[i] for i in idxs],
            next_raw_opes=[self.next_raw_opes[i] for i in idxs],
            next_raw_mas=[self.next_raw_mas[i] for i in idxs],
            next_proc_time=[self.next_proc_time[i] for i in idxs],
            next_ope_step_batch=[self.next_ope_step_batch[i] for i in idxs],
            next_mask_mas=[self.next_mask_ma[i] for i in idxs],
            next_mask_job_procing=[self.next_mask_job_procing[i] for i in idxs],
            next_mask_job_finish=[self.next_mask_job_finish[i] for i in idxs]
        )

    def _get_n_step_info(self, n_step_buffer: Deque, gamma: float):
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, done = n_step_buffer[-1][-2:]
        next_obs = n_step_buffer[-1][2]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, d = transition[-2:]
            n_o = transition[2]

            mask = ~d
            rew = r + gamma * rew * mask
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.ope_ma_adj)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
            self,
            max_size: int,
            batch_size: int = 32,
            alpha: float = 0.6,
            beta: float = 0.4,
            gamma: float = 0.95,
            n_step_horizon: int = 1,
            use_n_step: bool = False,
            to_cpu: bool = False
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(max_size, batch_size, gamma, n_step_horizon, use_n_step, to_cpu)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.beta = beta

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, transition):
        transition = super().store(transition)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert self.beta > 0

        idxs = self._sample_proportional()

        return dict(
            ope_ma_adj=itemgetter(*idxs)(self.ope_ma_adj),
            ope_pre_adj=itemgetter(*idxs)(self.ope_pre_adj),
            ope_sub_adj=itemgetter(*idxs)(self.ope_sub_adj),
            raw_opes=itemgetter(*idxs)(self.raw_opes),
            raw_mas=itemgetter(*idxs)(self.raw_mas),
            proc_time=itemgetter(*idxs)(self.proc_time),
            jobs_gather=itemgetter(*idxs)(self.jobs_gather),
            rewards=itemgetter(*idxs)(self.rewards),
            dones=itemgetter(*idxs)(self.is_terminals),
            actions=itemgetter(*idxs)(self.action_indexes),
            next_ope_ma_adj=itemgetter(*idxs)(self.next_ope_ma_adj),
            next_ope_pre_adj=itemgetter(*idxs)(self.next_ope_pre_adj),
            next_ope_sub_adj=itemgetter(*idxs)(self.next_ope_sub_adj),
            next_raw_opes=itemgetter(*idxs)(self.next_raw_opes),
            next_raw_mas=itemgetter(*idxs)(self.next_raw_mas),
            next_proc_time=itemgetter(*idxs)(self.next_proc_time),
            next_ope_step_batch=itemgetter(*idxs)(self.next_ope_step_batch),
            next_mask_mas=itemgetter(*idxs)(self.next_mask_ma),
            next_mask_job_procing=itemgetter(*idxs)(self.next_mask_job_procing),
            next_mask_job_finish=itemgetter(*idxs)(self.next_mask_job_finish),
            weights=np.array([self._calculate_weight(i) for i in idxs]),
            idxs=idxs
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-self.beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-self.beta)
        weight = weight / max_weight

        return weight
