import torch
import numpy as np
import pandas as pd
from crystal_design.utils.data_utils import build_crystal, build_crystal_dgl_graph, build_crystal_graph
from dgl.traversal import bfs_nodes_generator
import sys 
import mendeleev
import pickle
from tqdm import tqdm
import time
from copy import deepcopy
from torch.nn.functional import one_hot
import time
import warnings
warnings.simplefilter('ignore')

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
N_ATOMS_PEROV = 5

class OfflineTrajectories():
    def __init__(self, data, seed = 42, bfs_start = 0, sample_ind = 0, alpha = 1, beta = 0, reward_flag = True, graph_type = 'g'):
        self.data = data #pd.read_csv(file_name)
        self.n_vocab = len(ELEMENTS)
        self.sample_ind = sample_ind
        np.random.seed(seed)
        self.alpha = alpha
        self.beta = beta
        self.bfs_start = bfs_start
        self.reward_flag = reward_flag
        self.graph_type = graph_type
        if graph_type == 'g':
            self.state = self.random_initial_state()
        elif graph_type == 'mg':
            self.state = self.random_initial_state_mg()

    def random_initial_state_mg(self,):
        self.index = 0
        self.ret = 0
        cif_string = self.data.loc[self.sample_ind]['cif']
        canonical_crystal = build_crystal(cif_string)
        space_group_no = torch.tensor([canonical_crystal.get_space_group_info()[1]])
        state = build_crystal_graph(canonical_crystal, SPECIES_IND_INV)

        lengths = torch.tensor(canonical_crystal.lattice.abc)
        angles = torch.tensor(canonical_crystal.lattice.angles)
        state.lengths_angles_focus = torch.cat([lengths, angles, space_group_no])
        self.n_sites = state.num_nodes()
        self.history = []
        self.err_flag = 0

        if self.bfs_start >= self.n_sites:
            self.err_flag = 1
            return None
        self.traversal = torch.cat(list(bfs_nodes_generator(state, self.bfs_start)))
        try:
            assert len(self.traversal) == self.n_sites
        except:
            self.traversal = torch.tensor(list(range(self.n_sites)))
            self.err_flag = 1
        return state

    def random_initial_state(self):
        self.index = 0
        self.ret = 0
        cif_string = self.data.loc[self.sample_ind]['cif']
        canonical_crystal = build_crystal(cif_string)
        state = build_crystal_dgl_graph(canonical_crystal, SPECIES_IND_INV)
        lengths = torch.tensor(canonical_crystal.lattice.abc)
        angles = torch.tensor(canonical_crystal.lattice.angles)
        state.lengths_angles_focus = torch.cat([lengths, angles])
        self.n_sites = state.num_nodes()
        self.history = []
        self.traversal = torch.cat(list(bfs_nodes_generator(state, self.bfs_start)))
        self.err_flag = 0
        try:
            assert len(self.traversal) == self.n_sites
        except:
            self.traversal = torch.tensor(list(range(self.n_sites)))
            self.err_flag = 1
        return state

    def reset(self):
        self.state = self.random_initial_state()
    
    def ohe_to_atom_type(self, atom_types):
        atom_ind = torch.argmax(atom_types, dim = 1).tolist()
        atom_number = torch.tensor([SPECIES_IND[i] for i in atom_ind])
        return atom_number


    def step(self, eps = 0):
        """
        1. self.index stores the index
        2. index the node and update its atom type
        3. return updated state
        4. If index reached maximum value, terminate and assign reward
        self.state is a dgl graph object
        """
        done = False
        reward = 0
        if self.index < self.n_sites:
            node = self.traversal[self.index]
            r = np.random.rand()
            if r < 1 - eps:
                action = torch.where(self.state.ndata['true_atomic_number'][node])[0]
                if self.graph_type == 'mg':
                    action = self.state.ndata['true_atomic_number'][node]
            else:
                action = torch.random.choice(self.n_vocab)
                self.state = deepcopy(self.state)
            self.state.ndata['atomic_number'][node][action] = 1
            self.state.ndata['atomic_number'][node][-1] = 0
            if action == self.state.ndata['true_atomic_number'][node]:
                self.ret += 1
            reward = (0,None)
            self.index += 1 
            
            if self.index == self.n_sites:
                done = True
                frac_reward = self.ret / self.n_sites
                if self.reward_flag:
                    energy_reward = 0 
                else:
                    energy_reward = 0
                reward = (frac_reward, energy_reward)
        info = {}
        return (self.state, action, reward, done, None, info)

def run_episode(offline, eps = 0):
    obs = []
    if offline.index == 0:
        obs.append(deepcopy(offline.state))
    for i in range(offline.n_sites):
        # 0 - s1, a1
        # 1 - s2, a2
        # 2 - s3, a3
        # 3 - s4, a4
        # 4 - s5, a5
        obs[-1].lengths_angles_focus = torch.cat([obs[-1].lengths_angles_focus, one_hot(offline.traversal[offline.index], 5)])
        state, action, reward, done, _, info = offline.step(eps)
        obs[-1].action = torch.tensor(action)
        obs[-1].done = torch.tensor(done)
        obs.append(deepcopy(state))
    obs = obs[:-1]
    return obs, reward

def generate_data(file_name = '../data/mp_20/train.csv',
                  save_path = 'trajectories/train_Eformx5.pt'):
    trajectories_dict = {'data':[], 'rewards':[]}

    data = pd.read_csv(file_name)
    n_samples = data.shape[0]
    for i in tqdm(range(n_samples)):
        for j in range(N_ATOMS_PEROV):
            offline_setup = OfflineTrajectories(data = data, bfs_start = j, sample_ind = i, reward_flag = (j == 0), graph_type = 'mg')
            if offline_setup.err_flag == 1:
                break
            obs, reward = run_episode(offline_setup)
            if j == 0:
                trajectories_dict['rewards'].append(reward[-1])

            trajectories_dict['data'] += obs
    torch.save(trajectories_dict, save_path)


def ohe_to_atom_type(atom_types):
        atom_ind = torch.argmax(atom_types, dim = 1).tolist()
        atom_number = torch.tensor([SPECIES_IND[i] for i in atom_ind])
        return atom_number
    

def run_episode_tensor(graph_object, prop, eps):
    observations = []
    actions = []
    next_observations = []
    rewards = []
    bandgaps = []
    terminals = []
    traversal = graph_object.traversal
    n = graph_object.n_sites
    for i in range(n):
        d = {}
        node = traversal[i]
        r = np.random.rand()
        d['atomic_number'] = deepcopy(graph_object.state.ndata['atomic_number'])
        focus_feature = torch.zeros((n,1))
        focus_feature[node,0] = 1.
        d['atomic_number'] = torch.cat([d['atomic_number'], focus_feature], dim = 1)
        d['true_atomic_number'] = deepcopy(graph_object.state.ndata['true_atomic_number'])
        d['coordinates'] = deepcopy(graph_object.state.ndata['coords'])
        d['edges'] = graph_object.state.edges()
        d['etype'] = graph_object.state.edata['to_jimages']
        d['laf'] = graph_object.state.lengths_angles_focus #torch.cat([graph_object.state.lengths_angles_focus, one_hot(traversal[i], n)])
        d['ind'] = graph_object.sample_ind
        observations.append(d)
        if r < 1 - eps:
            action = torch.where(graph_object.state.ndata['true_atomic_number'][node])[0]
            if graph_object.graph_type == 'mg':
                action = graph_object.state.ndata['true_atomic_number'][node]
        else:
            action = torch.random.choice(graph_object.n_vocab)
            graph_object.state = deepcopy(graph_object.state)
        graph_object.state.ndata['atomic_number'][node][action] = 1
        graph_object.state.ndata['atomic_number'][node][-1] = 0
        d = {}
        d['atomic_number'] = deepcopy(graph_object.state.ndata['atomic_number'])
        if i < n-1:
            focus_feature = torch.zeros((n,1))
            focus_feature[traversal[i+1],0] = 1.
        else:
            focus_feature = torch.zeros((n,1))
        d['atomic_number'] = torch.cat([d['atomic_number'], focus_feature], dim = 1)
        d['true_atomic_number'] = deepcopy(graph_object.state.ndata['true_atomic_number'])
        d['coordinates'] = deepcopy(graph_object.state.ndata['coords'])
        d['edges'] = graph_object.state.edges()
        d['laf'] = graph_object.state.lengths_angles_focus
        d['ind'] = graph_object.sample_ind
        next_observations.append(d)
        actions.append(action)
        if i == n - 1:
            bg, energy = prop
            rewards.append(energy)
            bandgaps.append(bg)
            terminals.append(True)
        else:
            rewards.append(0.)
            terminals.append(False)

    return observations, actions, rewards, next_observations, bandgaps, terminals
    
def generate_trajectories_tensor(file_name, save_path, metals_prop, nonmetals_prop, eps = 0.):
    observations_list =  []
    actions_list =  []
    next_observations_list =  []
    rewards_list =  []
    terminals_list = []
    bandgaps_list = []
    data = pd.read_csv(file_name)
    s = list(nonmetals_prop[0].keys())
    j = 0
    for i in tqdm(s):
        if not nonmetals_prop[0].get(i,0):
            continue
        else:
            prop = nonmetals_prop[0][i][0], nonmetals_prop[1][i]
        if True:
            
            for j in range(N_ATOMS_PEROV):
                graph_object = OfflineTrajectories(data = data, bfs_start = j, sample_ind = i, reward_flag = (j == 0), graph_type = 'mg')
                if graph_object.err_flag == 1:
                    break
                try:
                    observations, actions, rewards, next_observations, bandgaps, terminals = run_episode_tensor(graph_object, prop, eps)
                except:
                    break
                observations_list += (observations)
                actions_list += (actions)
                next_observations_list += next_observations
                rewards_list += rewards
                terminals_list += terminals
                bandgaps_list += bandgaps
        
    d = {'observations': observations_list, 'actions': actions_list, 'next_observations': next_observations_list, 
        'rewards': rewards_list, 'bandgaps':bandgaps_list, 'terminals': terminals_list}
    print('Saving data')
    print('Number of crystals: ', len(rewards_list), len(terminals_list))
    torch.save(d, save_path)
    
if __name__ == '__main__':
    print('Started!')
    metals_prop = pickle.load(open('../files/TE_bg_dict_m.pkl', 'rb'))
    nonmetals_prop = pickle.load(open('../files/TE_bg_dict_nm.pkl', 'rb'))
    fe_dict_prop = pickle.load(open('../files/FE_dict_nm.pkl', 'rb'))
    generate_trajectories_tensor(file_name = '../data/mp_20/train.csv',
                                save_path = 'trajectories/train_Eformx5.pt',
                                metals_prop = metals_prop,
                                nonmetals_prop = (nonmetals_prop, fe_dict_prop))

