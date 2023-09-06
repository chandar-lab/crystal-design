# from gym.spaces import Discrete, Box
# from hive.envs import BaseEnv
# from hive.envs.env_spec import EnvSpec
import numpy as np
# import pymatgen.core.structure as S
# from pymatgen.analysis import energy_models as em
# import pymatgen.io.cif as cif
import pandas as pd 
import pickle
import mendeleev
# from crystal_design.utils.data_utils import build_crystal, build_crystal_dgl_graph
# from crystal_design.utils.converters.pyg_graph_to_tensor import PyGGraphToTensorConverter
# from crystal_design.utils.converters.compute_prop import Crystal, OptEval
from pymatgen.io.cif import CifWriter
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from ase.io import read
from ase.calculators.espresso import Espresso
import mendeleev
import dgl
import torch
import ase

# ELEMENTS = ['O', 'Tl', 'Co', 'N', 'Cr', 'Te', 'Sb', 'F', 'Ni', 'Pt', 'Ge', 'Y', 'S', 'Re', 'Rh', 'Ba', 'Bi', 'Cu', 'Mg', 'Ir', 'Al', 'Fe', 'Be', 'Ti', 'Nb', 'As', 'Sc', 'Cd', 'Sn', 'Li', 'Hf', 'Ga', 'Cs', 'Na', 'La', 'W', 'Si', 'In', 'Ca', 'Zn', 'Os', 'Hg', 'Zr', 'Sr', 'Ta', 'Mo', 'B', 'Mn', 'Au', 'Ag', 'K', 'V', 'Pb', 'Ru', 'Rb', 'Pd']
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

pseudo_dir = '/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/crystal-design/crystal_design/utils/SSSP'
pseudodict = pickle.load(open('/home/pragov/projects/rrg-bengioy-ad/pragov/crystal_structure_design/crystal-design/crystal_design/utils/pseudodict.pkl', 'rb'))
pwo_path = 'out'
nbnd = 256
nspin = 1
SI_BG = 4.0

input_data = {'prefix':"myprefix",'electron_maxstep':1000,'outdir':pwo_path,'pseudo_dir': pseudo_dir, 'tstress':False,'tprnfor':False,'calculation':'scf', 
                    'ecutrho':240,'verbosity':'high','ecutwfc':30, 'diagonalization': 'david', 'occupations':'fixed','smearing':'gaussian', 'mixing_mode':'plain', 
                    'mixing_beta':0.7,'degauss':0.001, 'nspin':nspin, 'nstep': 1, 'ntyp': 1, 'nbnd': nbnd}

class CrystalGraphEnvMP(object):
    
    def __init__(self, 
                file_name = None, 
                env_name = 'CrystalGraphEnvPerov', 
                config = {'max_num_nodes': 20, 'max_num_edges':25, 'node_ftr_dim': 88, 'to_numpy': False}, 
                coefs = {'alpha1':1, 'alpha2':5, 'beta':3},
                seed = 42, **kwargs):
        '''
        Crystal structure environment
        n_vocab : vocabulary size (number of elements in the action space)
        n_sites : number of sites in the unit cell
        species_ind : Index for elements
        atom_num_dict : Dictionary of atomic numbers
        file_name : CIF File containing crystal (temporary)
        env_name : name of environment
        seed : random seed value
        '''
        self.env_name = env_name
        self._seed = seed
        self.data = [pickle.load(open(file_name, 'rb'))]
        self.num_samples = len(self.data)
        self.init_vocab()
        # self.action_space = Discrete(self.n_vocab)
        # self.MAX_SIZE = config['max_num_nodes'] * config['node_ftr_dim'] * 2 + config['max_num_nodes'] * 3 + 2 * config['max_num_edges'] 
        # self.observation_space = Box(low = np.array([-np.inf] * self.MAX_SIZE), high = np.array([np.inf] * self.MAX_SIZE))
        self.coefs = coefs
        # self._env_spec = self.create_env_spec(self.env_name, **kwargs)
        # self.converter = PyGGraphToTensorConverter(config)
        self.state = self.random_initial_state()

    def to_graph(self, data_dict):
        atomic_number_list = data_dict['atomic_number']
        n_atoms = data_dict['atomic_number'].shape[0]
        true_atomic_number_list = data_dict['true_atomic_number']
        position_list = data_dict['coordinates']
        laf_list = torch.cat((data_dict['laf'], torch.tensor([SI_BG]))) #data_dict['laf'] #torch.cat((data_dict['laf'], torch.zeros(494-n_atoms)))
        edges_u_single =  data_dict['edges'][0] 
        edges_v_single =  data_dict['edges'][1] 
        edges_cat = torch.cat([edges_u_single[:,None], edges_v_single[:,None]], dim = 1)
        to_jimages = data_dict['etype']
        ind = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
        edges_cat = edges_cat[ind, :]
        edges_cat = torch.unique(edges_cat, dim = 0)
        edges_u = edges_cat[:,0]
        edges_v = edges_cat[:,1]    
        edata = torch.norm(position_list[edges_cat[:,0]] - position_list[edges_cat[:,1]], dim = 1)
        
        g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = n_atoms)#.to(device='cpu')
        g.ndata['atomic_number'] = atomic_number_list
        g.ndata['position'] = position_list
        g.ndata['true_atomic_number'] = true_atomic_number_list
        g.lengths_angles_focus = laf_list #torch.stack(laf_list)#.cuda()
        g.edata['e_feat'] = edata
        g = g.to(device = 'cuda')
        g.lengths_angles_focus = g.lengths_angles_focus.to(device = 'cuda')
        g.focus = data_dict['focus']
 
        return g


    def random_initial_state(self):
        '''
        Initialize state to default (for now, skeleton of one crystal)
        State --> [lattice, atom positions, coordinates, tracking pointer]
        To do : choose a random crystal skeleton from a collection
        '''

        self.index = 0
        self.ret = 0
        ind = np.random.choice(range(self.num_samples))
        observation = self.data[ind]
        crystal_graph = self.to_graph(observation)
        self.n_sites = crystal_graph.num_nodes()
        
        return crystal_graph
        
    def init_vocab(self):
        self.n_vocab = len(ELEMENTS)
        self.species_ind = {i:mendeleev.element(ELEMENTS[i]).atomic_number for i in range(self.n_vocab)}
        self.species_ind_inv = {mendeleev.element(ELEMENTS[i]).atomic_number:i for i in range(self.n_vocab)}

    # def create_env_spec(self, env_name, **kwargs):
    #     """
    #     Each family of environments have their own type of observations and actions.
    #     You can add support for more families here by modifying observation_space and action_space.
    #     """
    #     return EnvSpec(
    #         env_name=env_name,
    #         observation_space=[self.observation_space],
    #         action_space=[self.action_space],
    #     )

    def calc_reward(self):
        ase_obj = read(self.file_path)
        ase_obj.calc=Espresso(pseudopotentials=pseudodict,input_data=input_data, kpts=(3,3,3), label=pwo_path + '/espresso')
        ase_obj.get_total_energy()
        output_path = 'espresso.pwo'
        with open(pwo_path+'/'+output_path, 'r') as f:  
            try:
                lines = f.read()
                tmp = lines.split('highest occupied, lowest unoccupied level (ev):')[1].split()
                bg = float(tmp[1]) - float(tmp[0])
                if bg < 0.0:
                    bg = 0.0
                energy = float(lines.split('!    total energy              =')[1].split()[0])
                reward = self.coefs['alpha1'] * np.log10(-energy) + self.coefs['alpha2'] * np.exp(-(bg - SI_BG)**2 / self.coefs['beta'])
                return reward, bg, energy
            except:
                reward = None
                return reward, None, None
        
        


    def step(self, action):
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

            curr_focus = self.state.focus[self.index][0] ## Check

            self.state.ndata['atomic_number'][curr_focus][action] = 1.
            self.state.ndata['atomic_number'][curr_focus][-1] = 0.
            if self.index + 1 < self.n_sites:
                next_focus = self.state.focus[self.index+1][0]
                self.state.ndata['atomic_number'][next_focus][-1] = 1.
            else:
                self.state.ndata['atomic_number'][-1] = 0.
            if action == self.state.ndata['true_atomic_number'][self.index]:
                self.ret += 1
            reward = 0.
            bg = None
            energy = None
            self.index += 1 
        else:
            done = True
            self.file_path = 'tmp.cif'
            self.create_cif(self.state)
            reward, bg, energy = self.calc_reward()
            acc = self.ret / self.n_sites
        info = {}

        # self.state = self.converter.encode(state)
        return (self.state, reward, bg, energy, done)

    def create_cif(self, state):
        "Stores the constructed crystal as a .cif file in a temporary folder"
        n = self.n_sites

        ## GET ATOMIC NUMBERS AND CONVERT TO ATOM TYPES
        atomic_number = state.ndata['atomic_number'][:,:-2]
        atomic_number = torch.argmax(atomic_number, dim = 1)
        atom_types = [self.species_ind[int(atomic_number[i].cpu().numpy())] for i in range(atomic_number.shape[0])]

        ## GET POSITION
        coords = state.ndata['position']
        # GET STATE VARIABLES
        laf = state.lengths_angles_focus
        lengths = laf[:3].tolist()
        angles = laf[3:6].tolist()
        lattice_params = lengths + angles
        ## GET STRUCTURE
        canonical_crystal = Structure(lattice = Lattice.from_parameters(*lattice_params),
                                    species = atom_types, coords = coords.cpu(), coords_are_cartesian = True)

        ## WRITE TO CIF FILE
        writer = CifWriter(canonical_crystal)
        writer.write_file(self.file_path)   
    
    def reset(self):
        self.index = 0
        self.ret = 0
        self.state = self.random_initial_state()
        return self.state

    def seed(self, seed = 0):
        pass



