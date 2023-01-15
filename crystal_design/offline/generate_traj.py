import torch
import numpy as np
import pandas as pd
from crystal_design.utils.data_utils import build_crystal, build_crystal_dgl_graph
from crystal_design.utils.converters.compute_prop import Crystal, OptEval
from dgl.traversal import bfs_nodes_generator

class OfflineTrajectories():
    def __init__(self, file_name = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/data/perov_5/train.csv',
                seed = 42, sample_ind = 0):
        self.data = pd.read_csv(file_name)
        self.n_vocab = 56
        self.sample_ind = sample_ind
        np.random.seed()
    def random_initial_state(self):
        self.index = 0
        self.ret = 0
        # ind = np.random.choice(range(self.num_samples))
        cif_string = self.data.loc[self.sample_ind]['cif']
        canonical_crystal = build_crystal(cif_string)
        state = build_crystal_dgl_graph(canonical_crystal, self.species_ind_inv)
        self.n_sites = state.num_nodes()
        self.history = []
        self.traversal = list(bfs_nodes_generator(self.state, np.random.choice(self.n_sites)))
        assert len(self.traversal) == self.n_sites

    def reset(self):
        self.random_initial_state()
        
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
    for i in range(n_samples):
        offline_setup = OfflineTrajectories(index = i)
        trajectories_list.append(run_episode(offline_setup))
