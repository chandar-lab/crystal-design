import os
from typing import Any, Dict, List, Optional
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
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
from crystal_design.utils.utils import cart_to_frac_coords
from crystal_design.utils.compute_prop import Crystal
import mendeleev
from functools import partial
import yaml
from crystal_design.utils import to_cif

TensorBatch = List[torch.Tensor]
NUM_CLASSES = 88
PAD_SEQ = -10000


ELEMENTS = ['Cs', 'Er', 'Xe', 'Tc', 'Eu', 'Gd', 'Li', 'Hf', 'Dy', 'F', 'Te', 'Ti', 'Hg', 'Bi', 'Pr', 'Ne', 'Sm', 'Be', 'Au', 'Pb', 'C', 'Zr', 'Ir', 'Pd', 'Sc', 'Yb', 'Os', 'Nb', 'Ac', 'Rb', 'Al', 'P', 'Ga', 'Na', 'Cr', 'Ta', 'Br', 'Pu', 'Ge', 'Tb', 'La', 'Se', 'V', 'Pa', 'Ni', 'In', 'Cu', 'Fe', 'Co', 'Pm', 'N', 'K', 'Ca', 'Rh', 'B', 'Tm', 'I', 'Ho', 'Sb', 'As', 'Tl', 'Ru', 'U', 'Np', 'Cl', 'Re', 'Ag', 'Ba', 'H', 'O', 'Mg', 'W', 'Sn', 'Mo', 'Pt', 'Zn', 'Sr', 'S', 'Kr', 'Cd', 'Si', 'Y', 'Lu', 'Th', 'Nd', 'Mn', 'He', 'Ce']

TRANSITION_METALS = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
LANTHANIDES = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
ACTINIDES = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']
NOBLE = ['Xe', 'Ne', 'Kr', 'He']
HALOGENS = ['F', 'Br', 'Cl', 'I']
G1 = ['Li', 'Na', 'K', 'Rb', 'Cs']
G2 = ['Be', 'Mg', 'Ca', 'Sr', 'Ba']
NONMETALS = ['H','B', 'C', 'N', 'O', 'Si', 'P', 'S', 'As', 'Se', 'Te']
POST_TRANSITION_METALS = ['Al', 'Ga', 'Ge', 'In', 'Sn', 'Sb', 'Tl', 'Pb', 'Bi']


SPECIES_IND = {i:mendeleev.element(ELEMENTS[i]).atomic_number for i in range(len(ELEMENTS))}
SPECIES_IND_INV = {mendeleev.element(ELEMENTS[i]).atomic_number:i for i in range(len(ELEMENTS))}
N_ATOMS_PEROV = 1

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
    data_path: str = "crystal_design/offline/trajectories/train_Eformx5.pt"
    env: str = "crystal"  # environment name
    seed: int = 0  # random seed
    eval_freq: int = 10  # How often (time steps) we evaluate
    max_timesteps: int = int(2e5)   # Max time steps to run environment
    checkpoints_path: Optional[str] = 'models'  # Save path
    load_model: Optional[str] = ""  # Model load file name, "" doesn't load
    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 1024 #256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    qf_lr: float = 3e-4  # qnets learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_temp: float = 1.0  # CQL temperature
    cql_min_q_weight: float = 1.0  # Minimal Q weight
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    ## Added
    alpha1: float = 1.
    alpha2: float = 5.
    beta: float = 3.
    si_bg: float = 1.12
    # Wandb logging
    project: str = "CQL-NONMETALS"
    group: str = "tmp"
    name: str = "Run1"
    no_condition: bool = False
    bc: bool = False
    wandb_path: str = "wandb"

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
        self._focus_all = [torch.where(self._next_states_eval[i]['atomic_number'][:,-1])[0] for i in range(n_transitions)]   
        self._focus_all = [self._focus_all[i] if self._focus_all[i].size()[0] > 0 else torch.tensor(PAD_SEQ) for i in range(len(self._focus_all))] 
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def wandb_init(config: dict) -> None:
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
        beta: float = 3.,
        si_bg: float = 1.12,
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
        self.bc = bc
        self.total_it = step
        self.device = device

        self.qnet = qnet

        self.target_qnet = deepcopy(self.qnet).to(device)

        self.qnet_optimizer = qnet_optimizer
            
        if model_path != None:
            state_dict = torch.load(model_path)
            self.qnet.load_state_dict(state_dict['qnet1'])
            self.target_qnet.load_state_dict(state_dict['qnet1_target'])
            self.qnet_optimizer.load_state_dict(state_dict['qnet_optimizer'])

        self.si_bg = si_bg
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta

        self.total_it = step

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_qnet, self.qnet, soft_target_update_rate)

    def _q_loss(
        self, observations, actions, next_observations, rewards, bandgaps, dones, log_dict
    ):
        batch_size = observations.lengths_angles_focus.shape[0]
        q1_all = self.qnet(observations,observations.edata['e_feat'], observations.ndata['atomic_number'], observations.lengths_angles_focus)
        pred_actions = torch.argmax(q1_all, dim = 1)
        
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
        target = torch.ones_like(bandgaps) * self.si_bg
        target[zero_rew_ind] = 0.0
        alpha_2 = self.alpha2
        beta = self.beta
        alpha_1 = self.alpha1
        rewards = rewards * alpha_1 + alpha_2 * torch.exp(-(bandgaps - target)**2 / beta)
        rewards[zero_rew_ind] = 0.0
        td_target = rewards.unsqueeze(-1).to(device = self.device) + (1.0 - dones.to(dtype = torch.float32, device='cuda:0').unsqueeze(-1)) * self.discount * target_q_values
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
        qf_loss = cql_min_qf1_loss + 0.5 * qf1_loss 
        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                average_qf1=q1_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
                accuracy = acc,
                precision = prec,
            )
        )
        log_dict.update(
            dict(
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
        if policy == 'random':
            pred_actions = torch.randint(0, 88, (1,))
        else:
            q1_all = self.qnet(observations,observations.edata['e_feat'], observations.ndata['atomic_number'], observations.lengths_angles_focus)
            pred_actions = (torch.argmax(q1_all)) ### Nodes x 1 
        
        atomic_number = observations.ndata['atomic_number']   ### Nodes X (D + 2)
        curr_focus = atomic_number[:,-1]
        index_curr_focus = torch.where(curr_focus)[0]
        n = index_curr_focus.shape[0]
        try:
            t = pred_actions.shape[0]
        except:
            t = 1
        atomic_number[:, -2][index_curr_focus] = 0.  ## Correct
        try:
            atomic_number[index_curr_focus, pred_actions[t-n:]] = 1. ## Could lead to index mismatch issues
        except:
            atomic_number[index_curr_focus, pred_actions] = 1.
        next_focus = batch_focus[:,step]
        cum_sum_nsites = torch.cat((torch.tensor([0]), torch.cumsum(n_sites, dim = 0)[:-1]))
        next_focus = next_focus + cum_sum_nsites
        next_focus = next_focus[torch.where(next_focus >= 0)[0]]
        current_focus = torch.where(atomic_number[:, -1])[0] 
        atomic_number[:, -1][current_focus] = 0.
        atomic_number[:, -1][next_focus] = 1.
        observations.ndata['atomic_number'] = atomic_number

        return observations, pred_actions, index_curr_focus

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

        log_dict = {}
        observations = observations.to(device = self.device)
        observations.lengths_angles_focus = observations.lengths_angles_focus.to(device = self.device)
        next_observations = next_observations.to(device = self.device)
        next_observations.lengths_angles_focus = next_observations.lengths_angles_focus.to(device = self.device)
        """ Q function loss """
        if self.bc == True:
            qf_loss = self.supervised_loss(observations, actions, log_dict)
        else:
            qf_loss, pred_actions = self._q_loss(
                    observations, actions, next_observations, rewards, bandgaps.to(dtype = torch.float32), dones, log_dict
                )  ### Return a_max of q1_predicted
        

        self.qnet_optimizer.zero_grad()
        qf_loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=5., error_if_nonfinite = True)
        self.qnet_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qnet1": self.qnet.state_dict(),
            "qnet1_target": self.target_qnet.state_dict(),
            "qnet_optimizer": self.qnet_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qnet.load_state_dict(state_dict=state_dict["qnet1"])
        self.target_qnet.load_state_dict(state_dict=state_dict["qnet1_target"])
        self.qnet_optimizer.load_state_dict(
            state_dict=state_dict["qnet_optimizer"]
        )


def get_nsites(data):
    terminals = torch.tensor(data['terminals']).to(dtype = torch.int32)
    cum_sum_terminals = torch.where(terminals)[0]
    n_sites = torch.diff(cum_sum_terminals, prepend = torch.tensor([-1]))
    return n_sites

@pyrallis.wrap()
def train(config: TrainConfig, dict_ = None, model_path = None, step = 0):
    dataset = torch.load(config.data_path)
    replay_buffer = ReplayBuffer(   #Initializing replay buffer
        storage=ListStorage(max_size=config.buffer_size),
        batch_size = config.batch_size,
        collate_fn = partial(collate_function, si_bg = config.si_bg),
        pin_memory = True,
        prefetch = 32,
    )
    observations = dataset['observations']
    actions = dataset['actions']
    next_observations = dataset['next_observations']
    rewards = dataset['rewards']
    rewards = [torch.exp(-torch.tensor(r) / 5.) if r!= 0. else r for r in rewards]
    bandgaps = np.array([0.] * len(rewards))
    tmp_ind = np.where(np.array(dataset['rewards']))[0]
    bandgaps[tmp_ind] = np.array(dataset['bandgaps'])
    terminals = dataset['terminals']
    data = [(ob, a, nob, r, bg, done) for ob, a, nob, r, bg, done in zip(observations, actions, next_observations, rewards, bandgaps, terminals) ]
    index = replay_buffer.extend(data)
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed)

    qnet = MEGNetRL(no_condition = config.no_condition)
    qnet_optimizer = torch.optim.Adam(list(qnet.parameters()), config.qf_lr)

    
    kwargs = {
        "qnet": qnet,
        "qnet_optimizer": qnet_optimizer,
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
        'model_path': model_path,
        'step' : step,
        'alpha1':config.alpha1,
        'alpha2': config.alpha2,
        'beta': config.beta,
        'bc': config.bc
    }

    # Initialize actor
    trainer = CrystalCQL(**kwargs)  ##INITIALIZING CQL Instance
    wandb_init(asdict(config))
    for t in tqdm(range(step, int(config.max_timesteps))):
        batch = replay_buffer.sample()  ## SAMPLE FROM BUFFER
        log_dict = trainer.train(batch)  
        wandb.log(log_dict, step=trainer.total_it)
    
        if t % 25000 == 0 and config.checkpoints_path:
            torch.save(
                trainer.state_dict(),
                os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
            )
    return config

def ohe_to_atom_type(atom_types):
    atom_ind = torch.argmax(atom_types, dim = 1).tolist()
    atom_number = torch.tensor([SPECIES_IND[i] for i in atom_ind])
    return atom_number

@torch.no_grad()
@pyrallis.wrap()
def cql_eval(config: TrainConfig, model_path, data_path, save_path, cif_path, policy = None, si_bg = 1.12):
    replay_buffer = ReplayBufferCQL(   #Initializing replay buffer
        config.buffer_size,
        config.device,
    )
    categories = torch.load('../files/categories.pt')
    wandb_path = 'wandb/' + config.wandb_path
    metadata = yaml.safe_load(open(os.path.join(wandb_path , 'files/config.yaml'), 'r'))
    group = metadata['group']['value']
    seed = metadata['seed']['value']
    model_path = metadata['checkpoints_path']['value']
    state_dict = torch.load(os.path.join(model_path, 'checkpoint_0.pt'))
    si_bg = metadata['si_bg']['value']
    model = MEGNetRL(no_condition=False)
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
    trainer = CrystalCQL(**kwargs)
    data = torch.load(data_path)
    n_sites = get_nsites(data)
    replay_buffer.load_dataset(data, n_sites)
    loader_n = DataLoader(n_sites, batch_size = 1, num_workers = 0, shuffle = False)
    loader_focus = DataLoader(replay_buffer._focus_all, batch_size = 1, num_workers = 0, shuffle = False)
    eval_dataloader = DataLoader(replay_buffer.eval_dataset, batch_size = 1, collate_fn = partial(collate_function_eval, si_bg = si_bg), shuffle = False)
    recons_acc_list = []
    pred_accuracy_list = []
    cat_acc_list = []
    obs_list = []
    state_dict_list = [] 
    inds = []
    for batch, batch_focus, n_sites_batch in tqdm(zip(eval_dataloader, loader_focus, loader_n)):
        step = 0
        MAX_STEPS = batch_focus.shape[1]
        observations = batch #batch[0]
        for step in range(MAX_STEPS):
            observations, pred_actions, prev_focus = trainer.eval(observations, batch_focus, n_sites_batch, step, policy)
            true_actions = observations.ndata['true_atomic_number'][prev_focus]
        pred_types = torch.argmax(observations.ndata['atomic_number'], dim = 1)
        pred_cat = categories[pred_types.cpu(),1]
        true_types = observations.ndata['true_atomic_number']
        true_cat = categories[true_types.cpu(),1]
        matches = (pred_types == true_types)
        pred_acc = torch.mean(matches.float())
        cat_acc = torch.mean((true_cat == pred_cat).float())
        matches = torch.split(matches, n_sites_batch.tolist())
        pred_accuracy_list.append(pred_acc)
        cat_acc_list.append(cat_acc)
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
    

    #### UNCOMMENT FOR VALIDITY ####
    opt_crys = p_map(lambda x: Crystal(x), state_dict_list, num_cpus = 16)
    valid = [crys.comp_valid for crys in opt_crys]
    valid_inds = [inds[i] for i in range(len(inds)) if valid[i]]
    print('Group: ', group)
    print('Validity: ', sum(valid) / len(state_dict_list))
    obs_list = [obs_list[i] for i in range(len(obs_list)) if valid[i]]
    print(len(obs_list))

    print('Final recons accuracy = ', torch.mean(torch.tensor(recons_acc_list)))
    print('Final pred accuracy = ', torch.mean(torch.tensor(pred_accuracy_list)))
    print('Final Category Accuracy = ', torch.mean(torch.tensor(cat_acc_list)))
    save_path = os.path.join(save_path, group + f'_seed{seed}'  + '.pt')
    torch.save(obs_list, save_path)
    cif_path = os.path.join(cif_path, group + f'_seed{seed}')
    to_cif(data_path=save_path, save_path = cif_path)
    
if __name__ == "__main__":
    print('Starting')

    config = train()
    # data_path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/val_mp_nonmetals_x1.pt'
    # save_path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/runner/val_generated/nonmetals/eform' # + f"megnet-MG-w{config.cql_min_q_weight}-({config.alpha1}-{config.alpha2}-{config.beta})-{config.si_bg}eV_seed_{config.seed}.pt"
    # cif_path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/cifs_nm/new/eform' #+ f"VALmegnet-MG-w{config.cql_min_q_weight}-({config.alpha1}-{config.alpha2}-{config.beta})-{config.si_bg}eV_seed_{config.seed}"
    # cql_eval(model_path='', data_path=data_path, save_path = save_path, cif_path = cif_path, policy = None)

