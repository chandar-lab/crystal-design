import torch
import numpy as np
import pandas as pd
from crystal_design.utils.data_utils import build_crystal, build_crystal_dgl_graph
from crystal_design.utils.compute_prop import Crystal, OptEval
from dgl.traversal import bfs_nodes_generator
import sys 
import mendeleev
from tqdm import tqdm

ELEMENTS = ['O', 'Tl', 'Co', 'N', 'Cr', 'Te', 'Sb', 'F', 'Ni', 'Pt', 'Ge', 'Y', 'S', 'Re', 'Rh', 'Ba', 'Bi', 'Cu', 'Mg', 'Ir', 'Al', 'Fe', 'Be', 'Ti', 'Nb', 'As', 'Sc', 'Cd', 'Sn', 'Li', 'Hf', 'Ga', 'Cs', 'Na', 'La', 'W', 'Si', 'In', 'Ca', 'Zn', 'Os', 'Hg', 'Zr', 'Sr', 'Ta', 'Mo', 'B', 'Mn', 'Au', 'Ag', 'K', 'V', 'Pb', 'Ru', 'Rb', 'Pd']


class OfflineTrajectories():
    def __init__(self, file_name = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/perov_5/train.csv',
                seed = 42, sample_ind = 0):
        self.data = pd.read_csv(file_name)
        self.n_vocab = 56
        self.sample_ind = sample_ind
        np.random.seed()
        self.init_vocab()
        self.state = self.random_initial_state()
        
    def random_initial_state(self):
        self.index = 0
        self.ret = 0
        # ind = np.random.choice(range(self.num_samples))
        cif_string = self.data.loc[self.sample_ind]['cif']
        canonical_crystal = build_crystal(cif_string)
        state = build_crystal_dgl_graph(canonical_crystal, self.species_ind_inv)
        self.n_sites = state.num_nodes()
        self.history = []
        self.traversal = torch.cat(list(bfs_nodes_generator(state, np.random.choice(self.n_sites)))).numpy()
        assert len(self.traversal) == self.n_sites
        return state

    def init_vocab(self):
        self.n_vocab = len(ELEMENTS)
        self.species_ind = {i:mendeleev.element(ELEMENTS[i]).atomic_number for i in range(self.n_vocab)}
        self.species_ind_inv = {mendeleev.element(ELEMENTS[i]).atomic_number:i for i in range(self.n_vocab)}

    def reset(self):
        self.state = self.random_initial_state()
        
    def calc_prop(self, state):
        state_dict = {'frac_coords':state.ndata['position'], 'atom_types':state.ndata['atomic_number'], 'lengths':state.ndata['lengths'], 'angles':state.ndata['angles'], 'num_atoms':5}
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
        node = self.traversal[self.index]
        if self.index < self.n_sites:
            r = np.random.rand()
            if r < 1 - eps:
                action = torch.where(self.state.ndata['true_atomic_number'][node])[0]
            else:
                action = torch.random.choice(self.n_vocab)
            self.state.ndata['atomic_number'][node][action] = 1
            self.state.ndata['atomic_number'][node][-1] = 0
            if action == torch.where(self.state.ndata['true_atomic_number'][node])[0]:
                self.ret += 1
            reward = 0
            self.index += 1 
        else:
            done = True
            reward = self.ret / self.n_sites
        info = {}
        self.history.append((self.state, reward, done, None, info)) 
        return (self.state, reward, done, None, info)

def run_episode(offline):
    trajectories = []
    for i in range(offline.n_sites):
        trajectories.append(offline.step())
def generate_data(file_name = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/perov_5/train.csv'):
    trajectories_list = []
    data = pd.read_csv(file_name)
    n_samples = data.shape[0]
    for i in tqdm(range(n_samples)):
        offline_setup = OfflineTrajectories(sample_ind = i)
        trajectories_list.append(run_episode(offline_setup))
    torch.save(trajectories_list, '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/traj.pt')

if __name__ == '__main__':
    print('Started!')
    generate_data()
