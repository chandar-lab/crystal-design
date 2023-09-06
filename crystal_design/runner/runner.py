from copy import deepcopy
from crystal_design.agents.bc_agent import MEGNetRL
from crystal_design.envs import CrystalGraphEnvMP
from crystal_design.replays import ReplayBuffer
from torch import optim
from tqdm import tqdm
import torch
import wandb
class Runner():
    def __init__(self, data_path,
                num_episodes = 100,
                max_episode_length = 20,
                eps = 1e-1,
                batch_size = 128,
                discount = 0.99,
                tau = 5e-3):
        ## PENDING -- LOAD MODEL
        self.q_net = MEGNetRL()
        self.target_net = MEGNetRL()
        self.env = CrystalGraphEnvMP(file_name=data_path)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=5e-4, amsgrad=True)
        self.replay_buffer = ReplayBuffer(10000)
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.eps = eps
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau

    def run_episode(self):
        state = self.env.reset()
        tot_reward = 0.0
        for i in tqdm(range(self.max_episode_length)):
            action = self.select_action(state)
            next_state, reward, bg, energy, done = self.env.step(action.item())
            reward = torch.tensor([reward], device='cuda')
            if done:
                next_state = None
            self.replay_buffer.push(state, action, next_state, reward)
            tot_reward += reward
            state = deepcopy(next_state)
            self.optimize_model()
            self.update_target()
            if done:
                break
        return total_reward, bg, energy
        
    def train_agent(self):
        rewards = []
        bgs = []
        energies = []
        for step in range(self.num_episodes):
            total_reward, bg, energy = self.run_episode()
            rewards.append(total_reward)
            bgs.append(bg)
            energies.append(energy)
        return rewards, bgs, energies


    def select_action(self, state):
        Q_s_a = self.q_net(state, state.edata['e_feat'], state.ndata['atomic_number'], state.lengths_angles_focus)
        action = torch.argmax(Q_s_a)
        return action
    
    def update_target(self,):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.q_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        if len(self.replay_buffer.memory) < self.batch_size:
            return None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch, actions_batch, batch_next, rewards_batch, dones_batch = create_batch(transitions)
        Q_s_a = self.q_net(batch).gather(1, actions_batch)
        with torch.no_grad():
            Q_s_a_next = self.target_net(batch_next).max(1) * (1. - dones_batch)
        target = self.discount * Q_s_a_next + rewards_batch

        HuberLoss = nn.SmoothL1Loss()
        loss = HuberLoss(Q_s_a, target)
        
        self.optimizer.zero_grad()
        loss.backward()

        optimizer.step()



    
    
    