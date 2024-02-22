import os
from typing import Any, Dict, List, Optional
from copy import deepcopy
from dataclasses import asdict, dataclass
import random
import uuid
from crystal_design.agents import MEGNetRL
from crystal_design.utils import collate_function, collate_function_eval
from torcheval.metrics import MulticlassPrecision
import numpy as np
import pyrallis
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, ListStorage
from p_tqdm import p_map
import wandb
from tqdm import tqdm 
from crystal_design.utils import cart_to_frac_coords
from crystal_design.utils.compute_prop import Crystal
from functools import partial
import yaml
from crystal_design.utils import to_cif, create_sublists
from crystal_design.utils.variables import SPECIES_IND

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
TensorBatch = List[torch.Tensor]
NUM_CLASSES = 88
PAD_SEQ = -10000

@dataclass
class Config:
    # Experiment
    device: str = "cuda"
    data_path: str = "../offline/trajectories/train_Eformx5_small.pt" # path to trajectory data
    env: str = "crystal"  # environment name
    seed: int = 0  # random seed
    num_timesteps: int = int(2e5)   # Number of time steps
    checkpoints_path: Optional[str] = 'models'  # Save path
    model_path: Optional[str] = ""  # Model load file name, "" doesn't load
    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 1024 # Batch size (of transitions sampled from replay buffer)
    discount: float = 0.99  # Discount factor
    qf_lr: float = 3e-4  # qnets learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_temp: float = 1.0  # CQL temperature
    cql_min_q_weight: float = 1.0  # Minimal Q weight
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    ## Reward Design
    alpha1: float = 1.   # Design parameter of reward function
    alpha2: float = 10. # Design parameter of reward function
    beta1: float = 5. # Design parameter of reward function
    beta2: float = 3. # Design parameter of reward function
    p_hat: float = 1.12  # Target property
    no_condition: bool = False # True if model is not conditioned by property
    bc: bool = False # Behavioral cloning
    # Wandb logging
    project: str = "CQL-NONMETALS"  # Wandb Project name
    group: str = "tmp"  # Wandb Group name
    name: str = "Run1"  # Wandb Run name
    wandb_path: str = "wandb" # wandb folder path
    eval_wandb_path: str = '' 
    # Train / Eval Mode
    mode: str = "train" # 'train' or 'eval'

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data.cuda())


class ReplayBufferCQL:
    def __init__(
        self,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device

    def load_dataset(self, data: Dict[str, np.ndarray], n_sites):
        n_transitions = len(data["observations"])
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset"
            )
        init_indices = torch.cat((torch.tensor([0]), torch.cumsum(n_sites, dim = 0)[:-1]))
        self._states_eval = data["observations"]
        self._actions_eval = data["actions"]
        self._rewards_eval = data["rewards"]
        self._bandgaps = data["bandgaps"]
        self._next_states_eval = data["next_observations"]
        self._dones_eval = data["terminals"]
        self.eval_dataset = [[self._states_eval[i],self._actions_eval[i], self._next_states_eval[i],  self._rewards_eval[i], self._dones_eval[i]] for i in init_indices]
        self._focus_all = [self._next_states_eval[i]['atomic_number'][-1] for i in range(n_transitions)]   
        self._focus_all = [self._focus_all[i] if self._focus_all[i] < 20 else torch.tensor(PAD_SEQ) for i in range(len(self._focus_all))] 
        self._focus_all = create_sublists(self._focus_all, n_sites)
        self._focus_all = pad_sequence(self._focus_all, batch_first = True, padding_value = PAD_SEQ)
        self._pointer_eval = n_transitions
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._pointer, size=batch_size)
        states = [self._states[i] for i in indices]
        actions = [self._actions[i] for i in indices]
        rewards = [torch.log10(-torch.tensor(self._rewards[i])) if -self._rewards[i] > 0. else self._rewards[i] for i in indices] ###
        bandgaps = [self._bandgaps[i] for i in indices]
        next_states = [self._next_states[i] for i in indices]
        dones = [self._dones[i] for i in indices]
        states, actions, next_states, rewards, bandgaps, dones = collate_function_eval(states, actions, rewards, bandgaps, next_states, dones)
        return states, actions, next_states, rewards, bandgaps, dones 


    def add_transition(self):
        raise NotImplementedError


def set_seed(seed: int):
    """
    Random seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def wandb_init(config: dict) -> None:
    """
    Initialize wandb instance
    """
    wandb.init(
        config=config,
        resume=True,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


class CrystalCQL:
    def __init__(
        self,
        qnet,
        qnet_optimizer,
        discount: float = 0.99,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        target_update_period: int = 1,
        cql_temp: float = 1.0,
        cql_min_q_weight: float = 1.0,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
        model_path = None,
        step: int = 0,
        alpha1: float = 1.,
        alpha2: float = 5.,
        beta1: float = 5.,
        beta2: float = 3.,
        p_hat: float = 1.12,
        bc: bool = False
    ):
        super().__init__()

        self.discount = discount  ## DISCOUNT FACTOR
        self.qf_lr = qf_lr   # Q function learning rate
        self.soft_target_update_rate = soft_target_update_rate  # Soft target update rate
        self.target_update_period = target_update_period
        self.cql_temp = cql_temp
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device
        self.bc = bc # Behavioral Cloning mode
        self.total_it = step
        self.device = device

        self.qnet = qnet

        self.target_qnet = deepcopy(self.qnet).to(device)

        self.qnet_optimizer = qnet_optimizer
            
        if model_path is not None and os.path.isfile(model_path):
            state_dict = torch.load(model_path)
            self.qnet.load_state_dict(state_dict['qnet1'])
            self.target_qnet.load_state_dict(state_dict['qnet1_target'])
            self.qnet_optimizer.load_state_dict(state_dict['qnet_optimizer'])

        # Initialize reward design parameters and target value
        self.p_hat = p_hat
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2

        self.total_it = step

    def update_target_network(self, soft_target_update_rate: float):
        """
        Soft updating target network
        """
        soft_update(self.target_qnet, self.qnet, soft_target_update_rate)

    def _q_loss(
        self, observations, actions, next_observations, rewards, bandgaps, dones, log_dict
    ):
        batch_size = observations.lengths_angles_focus.shape[0]
        # Get Q values of all state-action pairs
        q1_all = self.qnet(observations,observations.edata['e_feat'], observations.ndata['atomic_number'], observations.lengths_angles_focus)
        pred_actions = torch.argmax(q1_all, dim = 1)
        
        # Q values of state-action pairs in the dataset
        q1_predicted = q1_all[range(batch_size), actions]
        actions = actions.cuda()
        assert pred_actions.shape == actions.shape

        prec_calc = MulticlassPrecision(num_classes = NUM_CLASSES, average = 'macro')
        prec = prec_calc.update(pred_actions.cpu(), actions.cpu()).compute()

        acc = torch.sum(pred_actions == actions).item() / batch_size

        with torch.no_grad():
            target_q_values = torch.max(self.target_qnet(next_observations, next_observations.edata['e_feat'], next_observations.ndata['atomic_number'], next_observations.lengths_angles_focus), dim = 1)[0]
                # Stop gradients here

        target_q_values = target_q_values.unsqueeze(-1)

        zero_rew_ind = torch.where(rewards == 0.)[0]
        target = torch.ones_like(bandgaps) * self.p_hat
        target[zero_rew_ind] = 0.0
        alpha_2 = self.alpha2
        beta = self.beta2
        alpha_1 = self.alpha1
        rewards = rewards * alpha_1 + alpha_2 * torch.exp(-(bandgaps - target)**2 / beta)
        rewards[zero_rew_ind] = 0.0
        td_target = rewards.unsqueeze(-1).to(device = self.device) + (1.0 - dones.to(dtype = torch.float32, device= self.device).unsqueeze(-1)) * self.discount * target_q_values
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())

        ## CQL
        cql_qf1_ood = torch.logsumexp(q1_all / self.cql_temp, dim = -1) * self.cql_temp

        # """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        
        cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
        # Compute total loss
        qf_loss = cql_min_qf1_loss + 0.5 * qf1_loss 

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                average_qf1=q1_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
                accuracy = acc,
                precision = prec,
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
            )
        )

        return qf_loss, pred_actions

    def supervised_loss(self, observations, actions, log_dict):
        """
        Behavioral Cloning mode
        """
        batch_size = observations.lengths_angles_focus.shape[0]
        q1_all = self.qnet(observations,observations.edata['e_feat'], observations.ndata['atomic_number'], observations.lengths_angles_focus)
        pred_actions = torch.argmax(q1_all, dim = 1)
        actions = actions.cuda()

        # Categorical Cross Entropy Loss
        loss = F.cross_entropy(q1_all, actions)

        assert pred_actions.shape == actions.shape
        prec_calc = MulticlassPrecision(num_classes = NUM_CLASSES, average = 'macro')
        prec = prec_calc.update(pred_actions.cpu(), actions.cpu()).compute()
        acc = torch.sum(pred_actions == actions).item() / batch_size

        log_dict.update(
            dict(
                qf1_loss=loss.item(),
                accuracy = acc,
                precision = prec,
            )
        )
        return loss

    def eval(self, observations, batch_focus, n_sites, step, policy = None):
 
        observations = observations.to(device = self.device)
        observations.lengths_angles_focus = observations.lengths_angles_focus.to(device = self.device)
        observations.focus = observations.focus.to(device = self.device)
        if policy == 'random':
            pred_actions = torch.randint(0, NUM_CLASSES, (1,))
        else:
            q1_all = self.qnet(observations,observations.edata['e_feat'], observations.ndata['atomic_number'], observations.lengths_angles_focus)
            pred_actions = (torch.argmax(q1_all)) ### Nodes x 1 
        
        atomic_number = deepcopy(observations.ndata['atomic_number'])   ### Nodes X (D + 2)
        index_curr_focus = observations.focus
        n = index_curr_focus.shape[0]
        try:
            t = pred_actions.shape[0]
        except:
            t = 1
        atomic_number[index_curr_focus] = pred_actions
        next_focus = batch_focus[:,step]
        observations.focus = next_focus.to(device = self.device)
        observations.ndata['atomic_number'] = atomic_number

        return observations, pred_actions, index_curr_focus

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        """
        Performs one gradient step with a batch of transitions
        Arguments
            batch: batch of transitions
        Returns
            log_dict: log dictionary
        """
        (
            observations,
            actions,
            next_observations,
            rewards,
            bandgaps,
            dones,
        ) = batch
        self.total_it += 1

        log_dict = {}
        
        # Transfer to device
        observations = observations.to(device = self.device)
        observations.focus = observations.focus.to(device = self.device)
        observations.lengths_angles_focus = observations.lengths_angles_focus.to(device = self.device)
        next_observations = next_observations.to(device = self.device)
        next_observations.lengths_angles_focus = next_observations.lengths_angles_focus.to(device = self.device)
        next_observations.focus = next_observations.focus.to(device = self.device)
        
        if self.bc == True: ## Behavioral Cloning
            qf_loss = self.supervised_loss(observations, actions, log_dict)
        else: ## CQL
            qf_loss, pred_actions = self._q_loss(
                    observations, actions, next_observations, rewards, bandgaps.to(dtype = torch.float32), dones, log_dict
                )  
        

        self.qnet_optimizer.zero_grad()
        qf_loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=5., error_if_nonfinite = True)
        self.qnet_optimizer.step()

        # Update Target
        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        """
        Return state_dict dictionary
        """
        return {
            "qnet1": self.qnet.state_dict(),
            "qnet1_target": self.target_qnet.state_dict(),
            "qnet_optimizer": self.qnet_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load model from state_dict
        """
        self.qnet.load_state_dict(state_dict=state_dict["qnet1"])
        self.target_qnet.load_state_dict(state_dict=state_dict["qnet1_target"])
        self.qnet_optimizer.load_state_dict(
            state_dict=state_dict["qnet_optimizer"]
        )


def get_nsites(data):
    """
    Returns the number of sites for each crystal given data of transitions
    """
    terminals = torch.tensor(data['terminals']).to(dtype = torch.int32)
    cum_sum_terminals = torch.where(terminals)[0]
    n_sites = torch.diff(cum_sum_terminals, prepend = torch.tensor([-1]))
    return n_sites

@pyrallis.wrap()
def train(config: Config, step = 0):
    """
    Train function: Trains model based on Config. 
    Performs evaluation if config.mode == 'eval'
    """
    # Evaluate if mode is eval
    if config.mode == 'eval':
        data_path = '../offline/trajectories/val_mp_nonmetals_x1_small.pt'
        save_path = 'val_generated' 
        cif_path = '../offline/cifs_gen'
        cql_eval(data_path=data_path, save_path = save_path, cif_path = cif_path, policy = None)
        return config
   
    # Load Dataset
    dataset = torch.load(config.data_path)
    
    #Initializing replay buffer
    replay_buffer = ReplayBuffer(   
        storage=ListStorage(max_size=config.buffer_size),
        batch_size = config.batch_size,
        collate_fn = partial(collate_function, p_hat = config.p_hat),
        pin_memory = True,
        prefetch = 32,
    )

    # Get all transitions
    observations = dataset['observations']
    actions = dataset['actions']
    next_observations = dataset['next_observations']
    rewards = dataset['rewards']
    rewards = [torch.exp(-torch.tensor(r) / config.beta1) if r!= 0. else r for r in rewards]
    bandgaps =  np.zeros(len(rewards)) 
    tmp_ind = np.where(np.array(dataset['rewards']))[0]
    bandgaps[tmp_ind] = np.array(dataset['bandgaps'])
    terminals = dataset['terminals']
    data = [(ob, a, nob, r, bg, done) for ob, a, nob, r, bg, done in zip(observations, actions, next_observations, rewards, bandgaps, terminals) ]
    replay_buffer.extend(data)

    # Checkpoint path
    if config.checkpoints_path is not None:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Random seed
    seed = config.seed
    set_seed(seed)

    # Initialize Q-Net and optimizer
    qnet = MEGNetRL(no_condition = config.no_condition)
    qnet_optimizer = torch.optim.Adam(list(qnet.parameters()), config.qf_lr)

    
    kwargs = {
        "qnet": qnet, # Q network
        "qnet_optimizer": qnet_optimizer, # Q network optimizer
        "discount": config.discount, # Discount factor
        "soft_target_update_rate": config.soft_target_update_rate, # Soft target update rate
        "device": config.device, # Device
        # CQL
        "qf_lr": config.qf_lr,
        "target_update_period": config.target_update_period,
        "cql_temp": config.cql_temp,
        "cql_min_q_weight": config.cql_min_q_weight,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
        'model_path': config.model_path,
        'bc': config.bc,
        'step' : step,
        # Reward
        'alpha1':config.alpha1,
        'alpha2': config.alpha2,
        'beta1': config.beta1,
        'beta2': config.beta2,
    }

    # Initialize CQL Instance
    trainer = CrystalCQL(**kwargs)  
    wandb_init(asdict(config))
    for t in tqdm(range(step, int(config.num_timesteps))):
        batch = replay_buffer.sample()  ## SAMPLE FROM BUFFER
        log_dict = trainer.train(batch)  
        wandb.log(log_dict, step=trainer.total_it) ## Log data to wandb
    
        if t % 25000 == 0 and config.checkpoints_path:
            torch.save(
                trainer.state_dict(),
                os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
            )
    return config

def ohe_to_atom_type(atom_types):
    atom_ind = atom_types.tolist()
    atom_number = torch.tensor([SPECIES_IND[i] for i in atom_ind])
    return atom_number

@torch.no_grad()
@pyrallis.wrap()
def cql_eval(config: Config, data_path, save_path, cif_path, policy = None):
    """
    Evaluation mode: The function loads a saved model and performs evaluation on validation data
    """
    # Initializing replay buffer
    replay_buffer = ReplayBufferCQL(   
        config.buffer_size,
        config.device,
    )
    # Load element categories
    categories = torch.load('../files/categories.pt')

    # If loading through wandb path
    if config.eval_wandb_path:
        eval_wandb_path = os.path.join(config.wandb_path ,config.eval_wandb_path)
        metadata = yaml.safe_load(open(os.path.join(eval_wandb_path , 'files/config.yaml'), 'r'))
        group = metadata['group']['value']
        seed = metadata['seed']['value']
        model_path = metadata['checkpoints_path']['value']
        state_dict = torch.load(os.path.join(model_path, 'checkpoint_250000.pt'))
        p_hat = metadata['p_hat']['value']

    # If loading through model path (checkpoint) directly
    elif config.model_path:
        state_dict = torch.load(config.model_path)
        group = config.group
        seed = config.seed
        p_hat = config.p_hat

    # Load QNetwork
    model = MEGNetRL(no_condition=config.no_condition)
    model.load_state_dict(state_dict['qnet1'])

    kwargs = {
        "qnet": model,
        "qnet_optimizer": torch.optim.Adam(list(model.parameters()), config.qf_lr),
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "qf_lr": config.qf_lr,
        "target_update_period": config.target_update_period,
        "cql_temp": config.cql_temp,
        "cql_min_q_weight": config.cql_min_q_weight,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    # Initialize trainer
    trainer = CrystalCQL(**kwargs)
    # Load validation data
    data = torch.load(data_path)
    # Number of sites
    n_sites = get_nsites(data)
    replay_buffer.load_dataset(data, n_sites) ## Replay buffer for data
    loader_n = DataLoader(n_sites, batch_size = 1, num_workers = 0, shuffle = False) ## Data loader for n_sites
    loader_focus = DataLoader(replay_buffer._focus_all, batch_size = 1, num_workers = 0, shuffle = False) ## Data loader for focus
    eval_dataloader = DataLoader(replay_buffer.eval_dataset, batch_size = 1, collate_fn = partial(collate_function_eval, p_hat = p_hat), shuffle = False) ## Data loader for transitions

    # Initialize lists
    recons_acc_list = []
    pred_accuracy_list = []
    cat_acc_list = []
    obs_list = []
    state_dict_list = [] 
    inds = []

    for batch, batch_focus, n_sites_batch in tqdm(zip(eval_dataloader, loader_focus, loader_n)):
        MAX_STEPS = batch_focus.shape[1]
        observations = batch 

        # Perform a rollout
        for step in range(MAX_STEPS):
            observations, pred_actions, prev_focus = trainer.eval(observations, batch_focus, n_sites_batch, step, policy)
            true_actions = observations.ndata['true_atomic_number'][prev_focus]
            if observations.focus == PAD_SEQ:
                break

        # Compute accuarcy, similarity, and exact reconstructions
        pred_types = observations.ndata['atomic_number']
        pred_cat = categories[pred_types.cpu(),1]
        true_types = observations.ndata['true_atomic_number']
        true_cat = categories[true_types.cpu(),1]
        matches = (pred_types == true_types)
        pred_acc = torch.mean(matches.float())
        cat_acc = torch.mean((true_cat == pred_cat).float())
        matches = torch.split(matches, n_sites_batch.tolist())
        pred_accuracy_list.append(pred_acc.item())
        cat_acc_list.append(cat_acc.item())
        matches = torch.tensor([torch.prod(mat) for mat in matches])
        recons_acc = torch.mean(matches.float())
        recons_acc_list.append(recons_acc)
        obs_list.append(observations)

        atomic_number = deepcopy(observations.ndata['atomic_number'])
        position = deepcopy(observations.ndata['position'])
        lengths = deepcopy(observations.lengths_angles_focus.cpu()[0][:3])
        angles = deepcopy(observations.lengths_angles_focus.cpu()[0][3:6])
        num_atoms = atomic_number.shape[0]
        frac_coords = cart_to_frac_coords(position.to(dtype=torch.float32).cpu(), lengths.unsqueeze(0), angles.unsqueeze(0), num_atoms)
        state_dict = {'frac_coords':np.array(frac_coords), 'atom_types':np.array(ohe_to_atom_type(atomic_number)), 'lengths':np.array(lengths), 'angles':np.array(angles), 'num_atoms':num_atoms}
        state_dict_list.append(state_dict)
        inds.append(batch.inds[0])
    

    # Compute Validity
    opt_crys = p_map(lambda x: Crystal(x), state_dict_list, num_cpus = 16)
    valid = [crys.comp_valid for crys in opt_crys]
    print('Group: ', group)
    print('Validity: ', sum(valid) / len(state_dict_list))
    obs_list = [obs_list[i] for i in range(len(obs_list)) if valid[i]]

    print('Final recons accuracy = ', np.mean(recons_acc_list))
    print('Final pred accuracy = ', np.mean(pred_accuracy_list))
    print('Final Category Accuracy = ', np.mean(cat_acc_list))

    os.makedirs(save_path, exist_ok = True)
    os.makedirs(cif_path, exist_ok = True)

    # Store generated crystals
    save_path = os.path.join(save_path, group + f'_seed{seed}'  + '.pt')
    torch.save(obs_list, save_path)
    cif_path = os.path.join(cif_path, group + f'_seed{seed}')
    # Convert to CIF files
    to_cif(data_path=save_path, save_path = cif_path)
    
if __name__ == "__main__":
    print('Starting')

    config = train()
    

