# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# STRONG UNDER-PERFORMANCE ON PART OF ANTMAZE TASKS. BUT IN IQL PAPER IT WORKS SOMEHOW
# https://arxiv.org/pdf/2006.04779.pdf
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid
from crystal_design.agents.bc_agent import EGNNAgentBC, RandomAgent, GCNAgentBC, LinearAgentBC
from crystal_design.utils import collate_function, collate_functionV2, collate_functionV3, collate_function_offline, collate_function_offline_eval
# import d4rl
import gym
import numpy as np
import pyrallis
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm 

TensorBatch = List[torch.Tensor]
PAD_SEQ = -10000
def create_sublists(A, B):
    output = []
    start_index = 0

    for num_items in B:
        sublist = torch.tensor(A[start_index:start_index+num_items.numpy()])
        output.append(sublist)
        start_index += num_items

    return output

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "crystal"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = 10 #int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/cql_models/models_1'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 1024 #256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 10.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    bc_steps: int = int(0)  # Number of BC steps at start
    target_update_period: int = 1  # Frequency of target nets updates
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_min_q_weight: float = 1000.0  # Minimal Q weight
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Wandb logging
    project: str = "CORL-Crystal"
    group: str = "CQL-Crystal-band-tmp-1024"
    name: str = "Run1"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        # self._states = torch.zeros(
        #     (buffer_size, state_dim), dtype=torch.float32, device=device
        # )
        # self._actions = torch.zeros(
        #     (buffer_size, action_dim), dtype=torch.float32, device=device
        # )
        # self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        # self._next_states = torch.zeros(
        #     (buffer_size, state_dim), dtype=torch.float32, device=device
        # )
        # self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
    

        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = len(data["observations"])
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states = data["observations"]
        self._actions = data["actions"]
        self._rewards = data["rewards"]
        n_sites = [self._states[i]['atomic_number'].shape[0] for i in range(len(self._states))]
        self._bandgaps = data["bandgaps"]
        self._bandgaps = [num for num, count in zip(self._bandgaps, n_sites) for _ in range(count-1)]
        self._next_states = data["next_observations"]
        self._dones = data["terminals"]
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")


    def load_eval_dataset(self, data: Dict[str, np.ndarray], n_sites):
        # if self._size != 0:
        #     raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = len(data["observations"])
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        init_indices = torch.cat((torch.tensor([0]), torch.cumsum(n_sites, dim = 0)[:-1]))
        self._states_eval = data["observations"]
        self._actions_eval = data["actions"]
        self._rewards_eval = data["rewards"]
        self._next_states_eval = data["next_observations"]
        self._dones_eval = data["terminals"]
        self.eval_dataset = [[self._states_eval[i],self._actions_eval[i], self._next_states_eval[i],  self._rewards_eval[i], self._dones_eval[i]] for i in init_indices]
        self._focus_all = [torch.where(self._next_states_eval[i]['atomic_number'][:,-1])[0] for i in range(n_transitions)]   ### ARGMAX MAY NOT BE CORRECT HERE
        self._focus_all = [self._focus_all[i] if self._focus_all[i].size()[0] > 0 else torch.tensor(PAD_SEQ) for i in range(len(self._focus_all))] 
        self._focus_all = create_sublists(self._focus_all, n_sites)
        self._focus_all = pad_sequence(self._focus_all, batch_first = True, padding_value = PAD_SEQ)
        self._pointer_eval = n_transitions
        print(f"Dataset size: {n_transitions}")

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states = data["observations"]
        self._actions = data["actions"]
        self._rewards = data["rewards"][..., None]
        self._next_states = data["next_observations"]
        self._dones = data["terminals"][..., None]
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = [self._states[i] for i in indices]
        actions = [self._actions[i] for i in indices]
        rewards = [-self._rewards[i] for i in indices]
        # n_sites = [self._states[i].shape[0] for i in indices]
        bandgaps = [self._bandgaps[i] for i in indices]
        next_states = [self._next_states[i] for i in indices]
        dones = [self._dones[i] for i in indices]
        states, actions, next_states, rewards, dones = collate_function_offline(states, actions, rewards, bandgaps, next_states, dones)
        return states, actions, next_states, rewards, dones #, indices, n_sites


    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)

@torch.no_grad()
def eval_actor_egnn(
    actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    np.randon.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class ContinuousCQL:
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_min_q_weight: float = 10.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount  ## DISCOUNT FACTOR
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning ## SOFT Q-LEARNING??
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr   # Policy learning rate
        self.qf_lr = qf_lr   # Q function learning rate
        self.soft_target_update_rate = soft_target_update_rate  # Soft target update rate
        self.bc_steps = bc_steps ## CHECK, behavioral cloning?
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions  ## ACTION SPACE?
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:  
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self, observations, actions, next_observations, rewards, dones, alpha, log_dict
    ):
    
        batch_size = observations.n_atoms.shape[0]
        q1_all = self.critic_1(observations)
        q2_all = self.critic_2(observations)
        pred_actions = torch.argmax(q1_all, dim = 1)
        
        q1_predicted = q1_all[range(batch_size), actions]
        q2_predicted = q2_all[range(batch_size), actions]
        actions = torch.stack(actions).cuda()
        assert pred_actions.shape == actions.shape
        acc = torch.sum(pred_actions == actions).item() / batch_size
        self.cql_max_target_backup = False
        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            # new_next_actions, next_log_pi = self.actor(next_observations)
            # target_q_values = torch.min(
            #     torch.max(self.target_critic_1(next_observations), dim = 1)[0],
            #     # torch.max(self.target_critic_2(next_observations), dim = 1)[0],
            # )   ## Add indexing here by indexing it with actions
            with torch.no_grad():
                target_q_values = torch.max(self.target_critic_1(next_observations), dim = 1)[0]
                ## Stop gradients here

        # if self.backup_entropy:
            # target_q_values = target_q_values - alpha * next_log_pi
        # target_q_values = torch.ones_like(q1_predicted).cuda() * 10000
        target_q_values = target_q_values.unsqueeze(-1)
        ### ADDED FOR CONSTANT REWARD ###
        # rewards[torch.where(rewards)] = 10000.
        td_target = rewards.unsqueeze(-1).to(device = 'cuda:0') + (1.0 - dones.to(dtype = torch.float32, device='cuda:0').unsqueeze(-1)) * self.discount * target_q_values
        td_target = td_target.squeeze(-1)
        # td_target = torch.ones_like(q1_predicted).cuda() * 10000
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        ## CQL
        # action_dim = actions[0].shape[-1]
        cql_qf1_ood = torch.logsumexp(q1_all / self.cql_temp, dim = -1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(q2_all / self.cql_temp, dim = -1) * self.cql_temp

        # cql_random_actions = actions.new_empty(   ##CQL_N_ACTIONS should sample discrete actions
        #     (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        # ).uniform_(-1, 1)  ## Change to discrete  Sample discrete actions
        # cql_current_actions, cql_current_log_pis = self.actor(
        #     observations, repeat=self.cql_n_actions   ## Check repeat argument - number of actions. In deterministic case?
        # )
        # cql_next_actions, cql_next_log_pis = self.actor(
        #     next_observations, repeat=self.cql_n_actions
        # )
        # cql_current_actions, cql_current_log_pis = (
        #     cql_current_actions.detach(),
        #     cql_current_log_pis.detach(),
        # )
        # cql_next_actions, cql_next_log_pis = (
        #     cql_next_actions.detach(),
        #     cql_next_log_pis.detach(),
        # )

        # cql_q1_rand = self.critic_1(observations, cql_random_actions)
        # cql_q2_rand = self.critic_2(observations, cql_random_actions)
        # cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        # cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        # cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        # cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        # cql_cat_q1 = torch.cat(
        #     [
        #         cql_q1_rand,
        #         torch.unsqueeze(q1_predicted, 1),
        #         cql_q1_next_actions,
        #         cql_q1_current_actions,
        #     ],
        #     dim=1,
        # )
        # cql_cat_q2 = torch.cat(
        #     [
        #         cql_q2_rand,
        #         torch.unsqueeze(q2_predicted, 1),
        #         cql_q2_next_actions,
        #         cql_q2_current_actions,
        #     ],
        #     dim=1,
        # )
        # cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        # cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        # if self.cql_importance_sample:
        #     random_density = np.log(0.5**action_dim)
        #     cql_cat_q1 = torch.cat(
        #         [
        #             cql_q1_rand - random_density,
        #             cql_q1_next_actions - cql_next_log_pis.detach(),
        #             cql_q1_current_actions - cql_current_log_pis.detach(),
        #         ],
        #         dim=1,
        #     )
        #     cql_cat_q2 = torch.cat(
        #         [
        #             cql_q2_rand - random_density,
        #             cql_q2_next_actions - cql_next_log_pis.detach(),
        #             cql_q2_current_actions - cql_current_log_pis.detach(),
        #         ],
        #         dim=1,
        #     )

        # cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        # cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        # """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime  # noqa
                * self.cql_min_q_weight  # noqa
                * (cql_qf1_diff - self.cql_target_action_gap)  # noqa
            )
            cql_min_qf2_loss = (
                alpha_prime  # noqa
                * self.cql_min_q_weight  # noqa
                * (cql_qf2_diff - self.cql_target_action_gap)  # noqa
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
            cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight
            # alpha_prime_loss = observations.new_tensor(0.0)
            # alpha_prime = observations.new_tensor(0.0)

        # qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss
        qf_loss = qf1_loss * 0.5 + cql_min_qf1_loss 

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                # qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                # average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
                accuracy = acc,
            )
        )
        log_dict.update(
            dict(
                # cql_std_q1=cql_std_q1.mean().item(),
                # cql_std_q2=cql_std_q2.mean().item(),
                # cql_q1_rand=cql_q1_rand.mean().item(),
                # cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                # cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                # cql_qf2_diff=cql_qf2_diff.mean().item(),
                # cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                # cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                # cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                # cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                # alpha_prime_loss=alpha_prime_loss.item(),
                # alpha_prime=alpha_prime.item(),
            )
        )
        alpha_prime = alpha_prime_loss = 0.
        return qf_loss, alpha_prime, alpha_prime_loss, pred_actions

    def eval(self, observations, batch_focus, n_sites, step):
        # (
        #     observations,
        #     actions,
        #     next_observations,
        #     rewards,
        #     dones,
        # ) = batch
        observations = observations.to(device = 'cuda:0')
        observations.lengths_angles_focus = observations.lengths_angles_focus.to(device = 'cuda:0')
        # next_observations = next_observations.to(device = 'cuda:0')
        # next_observations.lengths_angles_focus = next_observations.lengths_angles_focus.to(device = 'cuda:0')
        q1_all = self.critic_1(observations)
        pred_actions = torch.argmax(q1_all, dim = 1) ### Nodes x 1 
        atomic_number = observations.ndata['atomic_number']   ### Nodes X (D + 2)
        curr_focus = atomic_number[:,-1]
        index_curr_focus = torch.where(curr_focus)[0]
        n = index_curr_focus.shape[0]
        t = pred_actions.shape[0]
        atomic_number[:, -2][index_curr_focus] = 0.  ## Correct
        atomic_number[index_curr_focus, pred_actions[t-n:]] = 1. ## Could lead to index mismatch issues
        next_focus = batch_focus[:,step]
        cum_sum_nsites = torch.cat((torch.tensor([0]), torch.cumsum(n_sites, dim = 0)[:-1]))
        next_focus = next_focus + cum_sum_nsites
        next_focus = next_focus[torch.where(next_focus >= 0)[0]]
        current_focus = torch.where(atomic_number[:, -1])[0] 
        atomic_number[:, -1][current_focus] = 0.
        atomic_number[:, -1][next_focus] = 1.
        observations.ndata['atomic_number'] = atomic_number
        # batch[0] = observations

        return observations, pred_actions 

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            next_observations,
            rewards,
            bandgaps,
            dones,
        ) = batch
        self.total_it += 1


        # obs_graphs, actions, next_obs_graphs = collate_function(observations, next_observations)
        # new_actions, log_pi = self.actor(batch) ## Change actor to discrete for atom type prediction, with EGNN
                                                     ## Also, return actual action and log prob (try log softmax)
        # alpha, alpha_loss = self._alpha_and_alpha_loss(batch, log_pi)   ## Check again
        # alpha = 1.
        # """ Policy loss """
        # policy_loss = self._policy_loss(
        #     batch, new_actions, alpha, log_pi
        # )

        # log_dict = dict(
        #     log_pi=log_pi.mean().item(),
        #     policy_loss=policy_loss.item(),
        #     # alpha_loss=alpha_loss.item(),
        #     # alpha=alpha.item(),
        # )
        log_dict = {}
        observations = observations.to(device = 'cuda:0')
        observations.lengths_angles_focus = observations.lengths_angles_focus.to(device = 'cuda:0')
        next_observations = next_observations.to(device = 'cuda:0')
        next_observations.lengths_angles_focus = next_observations.lengths_angles_focus.to(device = 'cuda:0')
        alpha = torch.tensor(1.)
        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss, pred_actions = self._q_loss(
                observations, actions, next_observations, rewards, bandgaps, dones, alpha, log_dict
            )  ### Return a_max of q1_predicted
        
        ## Modify qf_loss and policy_loss, and also EGNNBC to include necessary methods and return values

        # if self.use_automatic_entropy_tuning:
        #     self.alpha_optimizer.zero_grad()
        #     # alpha_loss.backward()
        #     self.alpha_optimizer.step()

        # self.actor_optimizer.zero_grad()
        # policy_loss.backward()
        # self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True) 
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=5)
        self.critic_1_optimizer.step()
        # self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]

def get_nsites(data):
    terminals = torch.tensor(data['terminals']).to(dtype = torch.int32)
    cum_sum_terminals = torch.where(terminals)[0]
    n_sites = torch.diff(cum_sum_terminals, prepend = torch.tensor([-1]))
    return n_sites

@pyrallis.wrap()
def train(config: TrainConfig):
    # env = gym.make(config.env)
    env = None
    # state_dim = 118 #env.observation_space.shape[0]
    # action_dim = 1 #env.action_space.shape[0]

    dataset = torch.load('/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/train_mp_mg_3kbands.pt')
    n_sites = get_nsites(dataset)
    # if config.normalize_reward:
    #     modify_reward(dataset, config.env)

    # if config.normalize: ### Preprocessing state/rewards
    #     state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    # else:
    #     state_mean, state_std = 0, 1

    # dataset["observations"] = normalize_states(
    #     dataset["observations"], state_mean, state_std
    # )
    # dataset["next_observations"] = normalize_states(
    #     dataset["next_observations"], state_mean, state_std
    # )
    # env = wrap_env(env, state_mean=state_mean, state_std=state_std)  ## Check
    replay_buffer = ReplayBuffer(   #Initializing replay buffer
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_dataset(dataset) # Loading dataset into buffer
    # replay_buffer.load_eval_dataset(dataset, n_sites)
    # loader_focus = DataLoader(replay_buffer._focus_all, batch_size = 1024, num_workers = 0, shuffle = False)
    # eval_dataloader = DataLoader(replay_buffer.eval_dataset, batch_size = 256, collate_fn = collate_function_offline_eval, shuffle = False)
    # max_action = float(env.action_space.high[0]) # Check
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    # set_seed(seed, env)

    # critic_1 = FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init).to(
    #     config.device
    # )    ### REPLACE THIS WITH EGNN
    # critic_2 = FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init).to(
    #     config.device
    # )

    critic_1 = EGNNAgentBC(graph_type = 'mg')
    critic_2 = EGNNAgentBC(graph_type = 'mg')

    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    # actor = TanhGaussianPolicy(
    #     state_dim, action_dim, max_action, orthogonal_init=config.orthogonal_init
    # ).to(config.device)           ### REPLACE THIS WITH EGNN

    actor = EGNNAgentBC(graph_type = 'mg')
    

    # actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        # "target_entropy": -np.prod(env.action_space.shape).item(),
        "target_entropy": -1,
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_min_q_weight": config.cql_min_q_weight,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)  ##INITIALIZING CQL Instance
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in tqdm(range(int(config.max_timesteps))):
        batch = replay_buffer.sample(config.batch_size)  ## SAMPLE FROM BUFFER
        # batch = [b.to(config.device) for b in batch]  ## LOAD TO GPU
        log_dict = trainer.train(batch)   ## CHECK TRAIN FUNCTION in CQL CLASS
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        # if (t + 1) % config.eval_freq == 0:    ## EVALUATE on real env with trained actor
            # true_actions_list = replay_buffer._actions_eval
            # pred_actions_list = [] 
            # print(f"Time steps: {t + 1}")
            # for batch, batch_focus in zip(eval_dataloader, loader_focus):
            #     pred_actions, done = trainer.eval(batch, batch_focus, n_sites)
            #     pred_actions_list.append(pred_actions)
            # eval_scores = eval_actor(
            #     env,
            #     actor,
            #     device=config.device,
            #     n_episodes=config.n_episodes,
            #     seed=config.seed,
            # )   
            # eval_score = eval_scores.mean()
            # normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            # evaluations.append(normalized_eval_score)
            # print("---------------------------------------")
            # print(
            #     f"Evaluation over {config.n_episodes} episodes: "
            #     f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            # )
            # print("---------------------------------------")
        if t % 50 == 0 and config.checkpoints_path:
            torch.save(
                trainer.state_dict(),
                os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
            )
            # wandb.log(
            #     {"d4rl_normalized_score": normalized_eval_score},
            #     step=trainer.total_it,
            # )

@torch.no_grad()
@pyrallis.wrap()
def cql_eval(config: TrainConfig, model_path, data_path):
    replay_buffer = ReplayBuffer(   #Initializing replay buffer
        config.buffer_size,
        config.device,
    )
    state_dict = torch.load(model_path)
    model = EGNNAgentBC(graph_type = 'mg')
    model.load_state_dict(state_dict['critic1'])
    kwargs = {
        "critic_1": model,
        "critic_2": EGNNAgentBC(graph_type = 'mg'),
        "critic_1_optimizer": torch.optim.Adam(list(model.parameters()), config.qf_lr),
        "critic_2_optimizer": torch.optim.Adam(list(model.parameters()), config.qf_lr),
        "actor": EGNNAgentBC(graph_type = 'mg'),
        "actor_optimizer": torch.optim.Adam(list(model.parameters()), config.qf_lr),
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        # "target_entropy": -np.prod(env.action_space.shape).item(),
        "target_entropy": -1,
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_min_q_weight": config.cql_min_q_weight,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }
    trainer = ContinuousCQL(**kwargs)
    data = torch.load(data_path)
    n_sites = get_nsites(data)
    replay_buffer.load_eval_dataset(data, n_sites)
    loader_n = DataLoader(n_sites, batch_size = 13, num_workers = 0, shuffle = False)
    loader_focus = DataLoader(replay_buffer._focus_all, batch_size = 13, num_workers = 0, shuffle = False)
    eval_dataloader = DataLoader(replay_buffer.eval_dataset, batch_size = 13, collate_fn = collate_function_offline_eval, shuffle = False)
    recons_acc_list = []
    pred_accuracy_list = []
    for batch, batch_focus, n_sites_batch in tqdm(zip(eval_dataloader, loader_focus, loader_n)):
        step = 0
        MAX_STEPS = batch_focus.shape[1]
        observations = batch[0]
        for step in range(MAX_STEPS):
            observations, pred_actions = trainer.eval(observations, batch_focus, n_sites_batch, step)
            breakpoint()
        pred_types = torch.argmax(observations.ndata['atomic_number'], dim = 1)
        true_types = observations.ndata['true_atomic_number']
        matches = (pred_types == true_types)
        pred_acc = torch.mean(matches.float())
        matches = torch.split(matches, n_sites_batch.tolist())
        pred_accuracy_list.append(pred_acc)
        matches = torch.tensor([torch.prod(mat) for mat in matches])
        recons_acc = torch.mean(matches.float())
        recons_acc_list.append(recons_acc)
    print('Final recons accuracy = ', torch.mean(torch.tensor(recons_acc_list)))
    print('Final pred accuracy = ', torch.mean(torch.tensor(pred_accuracy_list)))
if __name__ == "__main__":
    train()
    # model_path = '../cql_models/models_1/Run1-crystal-4b5218b1/checkpoint_600000.pt'
    # data_path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/train_mp_mg_20.pt'
    # cql_eval(model_path=model_path, data_path=data_path)