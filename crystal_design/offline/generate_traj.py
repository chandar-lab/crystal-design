import torch
import numpy as np
import pandas as pd
from crystal_design.utils.data_utils import build_crystal, build_crystal_dgl_graph, build_crystal_graph
from crystal_design.utils.compute_prop import Crystal, OptEval, GenEval
from cdvae.common.data_utils import cart_to_frac_coords
from dgl.traversal import bfs_nodes_generator
import sys 
import mendeleev
import pickle
from tqdm import tqdm
import time
from copy import deepcopy
from torch.nn.functional import one_hot
from p_tqdm import p_map
import time
import pdb
import warnings
warnings.simplefilter('ignore')

# ELEMENTS = ['O', 'Tl', 'Co', 'N', 'Cr', 'Te', 'Sb', 'F', 'Ni', 'Pt', 'Ge', 'Y', 'S', 'Re', 'Rh', 'Ba', 'Bi', 'Cu', 'Mg', 'Ir', 'Al', 'Fe', 'Be', 'Ti', 'Nb', 'As', 'Sc', 'Cd', 'Sn', 'Li', 'Hf', 'Ga', 'Cs', 'Na', 'La', 'W', 'Si', 'In', 'Ca', 'Zn', 'Os', 'Hg', 'Zr', 'Sr', 'Ta', 'Mo', 'B', 'Mn', 'Au', 'Ag', 'K', 'V', 'Pb', 'Ru', 'Rb', 'Pd']
# ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
#                   'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 
#                   'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
#                   'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 
#                   'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 
#                   'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 
#                   'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 
#                   'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 
#                   'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 
#                   'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 
#                   'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 
#                   'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
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
        # ind = np.random.choice(range(self.num_samples))
        cif_string = self.data.loc[self.sample_ind]['cif']
        canonical_crystal = build_crystal(cif_string)
        state = build_crystal_dgl_graph(canonical_crystal, SPECIES_IND_INV)
        lengths = torch.tensor(canonical_crystal.lattice.abc)
        angles = torch.tensor(canonical_crystal.lattice.angles)
        state.lengths_angles_focus = torch.cat([lengths, angles])
        # state.angles = torch.tensor(canonical_crystal.lattice.angles)
        self.n_sites = state.num_nodes()
        self.history = []
        self.traversal = torch.cat(list(bfs_nodes_generator(state, self.bfs_start)))
        # self.traversal = torch.cat(list(bfs_nodes_generator(state, np.random.choice(self.n_sites))))
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
        
    def calc_prop(self, state):
        state_dict = {'frac_coords':state.ndata['frac_coords'], 'atom_types':self.ohe_to_atom_type(state.ndata['atomic_number']), 'lengths':state.lengths, 'angles':state.angles, 'num_atoms':5}
        crystal = Crystal(state_dict)
        opt_cal = OptEval([crystal])
        prop = opt_cal.get_metrics()[0]
        return -prop

    def step(self, eps = 0):
        """
        1. self.ind stores the index
        2. index the node and update its atom type
        3. return updated graph
        4. If index reached maximum value, compute fractional reward
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
                # breakpoint()
                # start=time.time()
                if self.reward_flag:
                    energy_reward = 0 #self.calc_prop(self.state)
                else:
                    energy_reward = 0
                reward = (frac_reward, energy_reward)
                # end = time.time()
                # print('Calculation time: ', end - start)
                # reward = self.alpha * frac_reward + self.beta * energy_reward
        # else:
        #     done = True
        #     frac_reward = self.ret / self.n_sites
        #     energy_reward = 0 # self.calc_prop(self.state)
        #     reward = self.alpha * frac_reward + self.beta * energy_reward
            # reward = self.ret / self.n_sites
        info = {}
        # self.history.append((self.state, reward, done, None, info))
        return (self.state, action, reward, done, None, info)

def run_episode(offline, eps = 0):
    # trajectories = []
    obs = []
    # acs = []
    # rews = []
    # ds = []
    if offline.index == 0:
        # offline.state.focus = offline.traversal[offline.index]
        obs.append(deepcopy(offline.state))
        # trajectories.append(offline.state)
    for i in range(offline.n_sites):
        # 0 - s1, a1
        # 1 - s2, a2
        # 2 - s3, a3
        # 3 - s4, a4
        # 4 - s5, a5
        obs[-1].lengths_angles_focus = torch.cat([obs[-1].lengths_angles_focus, one_hot(offline.traversal[offline.index], 5)])
        state, action, reward, done, _, info = offline.step(eps)
        obs[-1].action = torch.tensor(action)
        # obs[-1].reward = torch.tensor(reward)
        obs[-1].done = torch.tensor(done)
        # obs[-1].focus = offline.traversal[offline.index]
        # obs[-1].frac_reward = reward[0]
        # obs[-1].energy_reward = reward[1]
        obs.append(deepcopy(state))
        # acs.append(action)
        # rews.append(reward)
        # ds.append(done)
        # trajectories.append(offline.step(eps))
    obs = obs[:-1]
    return obs, reward #, acs, rews, ds #trajectories

def generate_data(file_name = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/perov_5/train.csv',
                  save_path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/traj_dict.pt'):
    # trajectories_list = []
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
        # trajectories_dict['rewards'] += [right_reward] * N_ATOMS_PEROV
        # break
        # trajectories_dict['actions'] += acs
        # trajectories_dict['rewards'] += rews
        # trajectories_dict['dones'] += ds
        # trajectories_list.append(run_episode(offline_setup))
    torch.save(trajectories_dict, save_path)

def calc_prop(state_dict):
    # state_dict = {'frac_coords':graph.ndata['frac_coords'], 'atom_types':graph.ohe_to_atom_type(graph.ndata['atomic_number']), 'lengths':graph.lengths, 'angles':graph.angles, 'num_atoms':5}
    crystal = Crystal(state_dict)
    opt_cal = OptEval([crystal])
    prop = opt_cal.get_metrics()[0]
    return prop

def ohe_to_atom_type(atom_types):
        atom_ind = torch.argmax(atom_types, dim = 1).tolist()
        atom_number = torch.tensor([SPECIES_IND[i] for i in atom_ind])
        return atom_number

def evaluate_test(data_path):
    data = torch.load(data_path)
    n = len(data)
    prop_list = []
    state_dict_list = []
    state_dict_true_list = []
    for batch in tqdm(data):
        batch = batch.to(device='cpu')
        batch_size = batch.action.shape[0]
        j = 0
        for i in (range(0,batch_size * 5, 5)):
            atomic_number = deepcopy(batch.ndata['atomic_number'][i:i+5,:])
            true_atomic_number = deepcopy(batch.ndata['true_atomic_number'][i:i+5,:])
            position = deepcopy(batch.ndata['position'][i:i+5,:])
            lengths = deepcopy(batch.lengths_angles_focus.cpu()[j,:3])
            angles = deepcopy(batch.lengths_angles_focus.cpu()[j,3:6])
            # true_atomic_number = deepcopy(batch.ndata['true_atomic_number'][i:i+5,:])
            j += 1
            num_atoms = 5
            frac_coords = cart_to_frac_coords(position.to(dtype=torch.float32), lengths.unsqueeze(0), angles.unsqueeze(0), num_atoms)
            state_dict = {'frac_coords':frac_coords, 'atom_types':ohe_to_atom_type(atomic_number), 'lengths':lengths, 'angles':angles, 'num_atoms':num_atoms}
            state_dict_true = {'frac_coords':frac_coords, 'atom_types':ohe_to_atom_type(true_atomic_number), 'lengths':lengths, 'angles':angles, 'num_atoms':num_atoms}
            state_dict_true_list.append(state_dict_true)
            state_dict_list.append(state_dict)
            # prop = calc_prop(state_dict)
            # prop_list.append(prop)
    opt_crys = p_map(lambda x: Crystal(x), state_dict_list, num_cpus = 16)
    # gt_crys = p_map(lambda x: Crystal(x), state_dict_true_list, num_cpus = 16)
    gen_evaluator = GenEval(opt_crys)#, gt_crys)
    # opt_cal = OptEval(opt_crys)
    # prop_list, metrics = opt_cal.get_metrics()
    gen_metrics = gen_evaluator.get_metrics()
    print('Gen metrics: ', gen_metrics)
    print('Validity: ', len(prop_list) / len(state_dict_list))
    # print('Metrics: ', metrics)
    torch.save(prop_list, 'prop_list_test_final.pt') 
    return gen_metrics      

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
        # atomic_number = deepcopy(d['atomic_number'])
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
        # d['etype'] = graph_object.state.edata['to_jimages']
        d['laf'] = graph_object.state.lengths_angles_focus
        d['ind'] = graph_object.sample_ind
        # if i < n-1:
        #     d['laf'] = torch.cat([graph_object.state.lengths_angles_focus, one_hot(traversal[i + 1], n)])
        # else:
        #     d['laf'] = torch.cat([graph_object.state.lengths_angles_focus, torch.zeros(n)])
        next_observations.append(d)
        actions.append(action)
        if i == n - 1:
            bg, energy = prop
            # reward = None
            # bandgap = None
            # try:
            #     with open(reward_path, 'r') as file:
            #         lines = file.read()
            #         high, low = lines.split('highest occupied, lowest unoccupied level (ev):')[1].split('!')[0].split()
            #         bandgap = float(low)-float(high)
            #         if bandgap <0:
            #             bandgap = 0
            #         reward = float(lines.split('!    total energy')[1].split()[1])
            #             # rewards.append(reward)
            #             # true.append(data.loc[i]['bandgap'])

            # except:
            #     pass
                # for line in file:
                #     if '!    total energy' in line:
                #         reward = float(line.split()[4])
                #         break
                #     elif 'total energy' in line: 
                #         if 'following' not in line:
                #             reward = float(line.split()[-2])
            # assert reward != None and bandgap != None
            rewards.append(energy)
            bandgaps.append(bg)
            terminals.append(True)
            # print(reward)
        else:
            rewards.append(0.)
            terminals.append(False)
        # if action == graph_object.state.ndata['true_atomic_number'][node]:
        #     ret += 1
    # observations.append(d)

    return observations, actions, rewards, next_observations, bandgaps, terminals
def generate_trajectories_tensor(file_name, save_path, metals_prop, nonmetals_prop, eps = 0.):
    observations_list =  []
    actions_list =  []
    next_observations_list =  []
    rewards_list =  []
    terminals_list = []
    bandgaps_list = []
    data = pd.read_csv(file_name)
    s = list(metals_prop.keys()) + list(nonmetals_prop.keys())
    n_samples = data.shape[0]
    for i in tqdm(s):
        flag = 0
        # reward_path = reward_dir + 'espresso_' + str(i) + '.pwo'
        if not metals_prop.get(i,0) and not nonmetals_prop.get(i,0):
            continue
        elif metals_prop.get(i,0):
            prop = metals_prop[i]
        else:
            prop = nonmetals_prop[i]
        for j in range(N_ATOMS_PEROV):
            graph_object = OfflineTrajectories(data = data, bfs_start = j, sample_ind = i, reward_flag = (j == 0), graph_type = 'mg')
            if graph_object.err_flag == 1:
                break
            try:
                observations, actions, rewards, next_observations, bandgaps, terminals = run_episode_tensor(graph_object, prop, eps)
            except:
                flag = 1
                break
            observations_list += (observations)
            # print(len(observations_list))
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
    metals_prop = pickle.load(open('m_dict.pkl', 'rb'))
    nonmetals_prop = pickle.load(open('nm_dict.pkl', 'rb'))
    generate_trajectories_tensor(file_name = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/mp_20/train.csv',
                                save_path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/train_mp_mg_x1.pt',
                                metals_prop = metals_prop,
                                nonmetals_prop = nonmetals_prop)

    # for traj in [5]:
    #     test_valid_list = []
    #     for run in range(4,5): 
    #         path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/runner/random_agent_' + str(run)#'/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/models/perov_bc/GCNAgentBC_bc_11kx'+str(traj)+'_2048b0.001lr1e-5wdadam_4layer_run' + str(run) + '/'
    #         gen_metrics = evaluate_test(data_path=path +'bc_gen.pt')
    #         test_valid_list.append(gen_metrics['comp_valid'])
    #     print(str(traj) + ' Traj Test Validity: ', np.mean(test_valid_list))
