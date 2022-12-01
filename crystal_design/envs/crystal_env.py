from gym.spaces import Discrete, Box
from hive.envs import BaseEnv
from hive.envs.env_spec import EnvSpec
import numpy as np
import pymatgen.core.structure as S
from pymatgen.analysis import energy_models as em
import pymatgen.io.cif as cif

class CrystalEnv(BaseEnv):
    def __init__(self, n_vocab = 4, n_sites = 48, species_ind = {0:'Cu', 1:'P', 2:'N', 3:'O'},  #{0:'Cu', 1:'P', 2:'N', 3:'O'}, 
                 atom_num_dict = {}, file_name = '/network/scratch/p/prashant.govindarajan/crystal_design_project/code/Cu3P2NO6.cif', env_name = 'CrystalEnv', seed = 42, **kwargs):
        """
        Crystal structure environment
        n_vocab : vocabulary size (number of elements in the action space)
        n_sites : number of sites in the unit cell
        species_ind : Index for elements
        atom_num_dict : Dictionary of atomic numbers
        file_name : CIF File containing crystal (temporary)
        env_name : name of environment
        seed : random seed value
        """
        
        self.env_name = env_name
        self.n_vocab = n_vocab
        # Size of state vector
        self.state_size = 9 + (5 + n_vocab) * n_sites  ## 9 for lattice, 1 + n_vocab for each element and blank space, 3 for coordinates, n_sites for tracking position
        self.species_ind = species_ind
        self.atom_num_dict = atom_num_dict
        self.n_sites = n_sites
        self.t = 0
        self._seed = seed
        self.file_name = file_name

        #Action space
        self.action_space = Discrete(self.n_vocab)
        # State space
        self.observation_space = Box(low = np.array([-np.inf] * self.state_size), high = np.array([np.inf] * self.state_size))

        self._env_spec = self.create_env_spec(self.env_name, **kwargs)

        ### Temporary ###
        # Parse CIF File
        self.mat =  cif.CifParser(self.file_name).as_dict()['Cu3P2NO6']
        # Get lattice vector
        self.lattice = cif.CifParser(self.file_name).get_lattice(self.mat)
        self.lat_mat = np.ravel(self.lattice.matrix)
        # Initialize state
        self.state = self.random_initial_state()
        # print(self.state)
        
        self.ele = self.mat['_atom_site_type_symbol']
        
        ### Test ###
        self.ret = 0

    def random_initial_state(self):
        """
        Initialize state to default (for now, skeleton of one crystal)
        State --> [lattice, atom positions, coordinates, tracking pointer]
        To do : choose a random crystal skeleton from a collection
        """

        ele = self.mat['_atom_site_type_symbol']

        coords_x = self.mat['_atom_site_fract_x']
        coords_y = self.mat['_atom_site_fract_y']
        coords_z = self.mat['_atom_site_fract_z']

        state = np.array([])

        for i in range(self.n_sites):
            tmp = np.zeros(self.n_vocab + 1)
            tmp[-1] = 1
            c = np.array([float(coords_x[i]), float(coords_y[i]), float(coords_z[i])])
            state = np.concatenate([state, tmp, c])
        pointer = np.zeros(self.n_sites)
        pointer[0] = 1
        state = np.concatenate([self.lat_mat, state, pointer])

        return state

    def calc_energy(self, state):
        """
        Calculate energy of material using SymmetryMode (Pymatgen)
        state : representation of state
        returns : Energy calculated by Symmetry Model
        """
        ele = []
        coords = []
        for i in range(9, self.state_size - self.n_sites, self.n_vocab + 4):
            predictions = state[i : i + self.n_vocab]
            # print(predicitons)
            positions = list(state[i + self.n_vocab + 1 : i + self.n_vocab + 4])
            index = np.where(predictions)[0][0]
            ele.append(self.species_ind[index])
            coords.append(positions)
        struct = S.Structure(lattice = self.lattice, species = ele, coords = coords)
        energy = em.SymmetryModel().get_energy(struct)
        return energy

    def calc_reward(self, energy, thresh = 0):
        """
        Calculate reward using energy
        """
        ### trial : reward = energy for the timebeing
        reward = energy
        ###
        
        return reward

    def proxy_reward(self, state):
        true_ele = self.mat['_atom_site_type_symbol']
        pred_ele = []
        reward = 0
        k = 0
        for i in range(9, self.state_size - self.n_sites, self.n_vocab + 4):
            predictions = state[i : i + self.n_vocab]
#             print(predictions, len(predictions))
            index = np.where(predictions)[0][0]
            reward += int(true_ele[k] == self.species_ind[index])
            k += 1
        return reward

    def create_env_spec(self, env_name, **kwargs):
        """
        Each family of environments have their own type of observations and actions.
        You can add support for more families here by modifying observation_space and action_space.
        """
        return EnvSpec(
            env_name=env_name,
            observation_space=[self.observation_space],
            action_space=[self.action_space],
        )
    def reset(self):
        self.t = 0
        self.ret = 0
        self.state = self.random_initial_state()
        return self.state, None

    def step(self, action):
        """
        step function
        action : action chosen by the agent (element)
        """

        done = False
        if self.t < self.n_sites:
            # Moving pointer by one position
            pos1 = 9 + (self.n_vocab + 1 + 3) * self.n_sites + self.t  
            self.state[pos1] = 0
            try:
                self.state[pos1 + 1] = 1
            except:
                pass
            # Updating OHE of element type based on action
            pos2 = 9 + (self.n_vocab + 1 + 3) * self.t 
            self.state[pos2 : pos2 + self.n_vocab + 1][action] = 1
            self.state[pos2 : pos2 + self.n_vocab + 1][-1] = 0
            reward = 0
            self.t += 1
#             print(self.species_ind[action], end = ', ')
            if self.species_ind[action] == self.ele[self.t - 1]:
#                 reward = 1
                self.ret += 1
            else:
                reward = 0
#             reward = 0
        else:
            done = True
            # reward = self.calc_reward(self.calc_energy(self.state))
#             reward = self.proxy_reward(self.state)
            reward = 0
            reward = self.ret
#             print()
#             print(reward)


        info = {}
        # print(self.state)
        return (
            self.state, 
            reward,
            done,
            None,
            info
        )
    def seed(self, seed = 0):
        """
        random seed
        """
        self._seed = seed
    def close(self):
        pass





class CrystalEnvV2(BaseEnv):
    def __init__(self, n_vocab = 4, n_sites = 48, species_ind = {0:'Cu', 1:'P', 2:'N', 3:'O'},  #{0:'Cu', 1:'P', 2:'N', 3:'O'}, 
                 atom_num_dict = {}, file_name = '/network/scratch/p/prashant.govindarajan/crystal_design_project/code/Cu3P2NO6.cif', env_name = 'CrystalEnvV2', seed = 42, **kwargs):
        """
        Crystal structure environment without lattice information
        n_vocab : vocabulary size (number of elements in the action space)
        n_sites : number of sites in the unit cell
        species_ind : Index for elements
        atom_num_dict : Dictionary of atomic numbers
        file_name : CIF File containing crystal (temporary)
        env_name : name of environment
        seed : random seed value
        """
        
        self.env_name = env_name
        self.n_vocab = n_vocab
        # Size of state vector
        self.state_size = (5 + n_vocab) * n_sites  ## 9 for lattice, 1 + n_vocab for each element and blank space, 3 for coordinates, n_sites for tracking position
        self.species_ind = species_ind
        self.atom_num_dict = atom_num_dict
        self.n_sites = n_sites
        self.t = 0
        self._seed = seed
        self.file_name = file_name

        #Action space
        self.action_space = Discrete(self.n_vocab)
        # State space
        self.observation_space = Box(low = np.array([-np.inf] * self.state_size), high = np.array([np.inf] * self.state_size))

        self._env_spec = self.create_env_spec(self.env_name, **kwargs)

        ### Temporary ###
        # Parse CIF File
        self.mat =  cif.CifParser(self.file_name).as_dict()['Cu3P2NO6']
        # Get lattice vector
        self.lattice = cif.CifParser(self.file_name).get_lattice(self.mat)
        self.lat_mat = np.ravel(self.lattice.matrix)
        # Initialize state
        self.state = self.random_initial_state()
        # print(self.state)
        
        self.ele = self.mat['_atom_site_type_symbol']
        
        ### Test ###
        self.ret = 0

    def random_initial_state(self):
        """
        Initialize state to default (for now, skeleton of one crystal)
        State --> [lattice, atom positions, coordinates, tracking pointer]
        To do : choose a random crystal skeleton from a collection
        """

        ele = self.mat['_atom_site_type_symbol']

        coords_x = self.mat['_atom_site_fract_x']
        coords_y = self.mat['_atom_site_fract_y']
        coords_z = self.mat['_atom_site_fract_z']

        state = np.array([])

        for i in range(self.n_sites):
            tmp = np.zeros(self.n_vocab + 1)
            tmp[-1] = 1
            c = np.array([float(coords_x[i]), float(coords_y[i]), float(coords_z[i])])
            state = np.concatenate([state, tmp, c])
        pointer = np.zeros(self.n_sites)
        pointer[0] = 1
        state = np.concatenate([state, pointer])

        return state

    def calc_energy(self, state):
        """
        Calculate energy of material using SymmetryMode (Pymatgen)
        state : representation of state
        returns : Energy calculated by Symmetry Model
        """
        ele = []
        coords = []
        for i in range(9, self.state_size - self.n_sites, self.n_vocab + 4):
            predictions = state[i : i + self.n_vocab]
            # print(predicitons)
            positions = list(state[i + self.n_vocab + 1 : i + self.n_vocab + 4])
            index = np.where(predictions)[0][0]
            ele.append(self.species_ind[index])
            coords.append(positions)
        struct = S.Structure(lattice = self.lattice, species = ele, coords = coords)
        energy = em.SymmetryModel().get_energy(struct)
        return energy

    def calc_reward(self, energy, thresh = 0):
        """
        Calculate reward using energy
        """
        ### trial : reward = energy for the timebeing
        reward = energy
        ###
        
        return reward

    def proxy_reward(self, state):
        true_ele = self.mat['_atom_site_type_symbol']
        pred_ele = []
        reward = 0
        k = 0
        for i in range(9, self.state_size - self.n_sites, self.n_vocab + 4):
            predictions = state[i : i + self.n_vocab]
#             print(predictions, len(predictions))
            index = np.where(predictions)[0][0]
            reward += int(true_ele[k] == self.species_ind[index])
            k += 1
        return reward

    def create_env_spec(self, env_name, **kwargs):
        """
        Each family of environments have their own type of observations and actions.
        You can add support for more families here by modifying observation_space and action_space.
        """
        return EnvSpec(
            env_name=env_name,
            observation_space=[self.observation_space],
            action_space=[self.action_space],
        )
    def reset(self):
        self.t = 0
        self.ret = 0
        self.state = self.random_initial_state()
        return self.state, None

    def step(self, action):
        """
        step function
        action : action chosen by the agent (element)
        """

        done = False
        if self.t < self.n_sites:
            # Moving pointer by one position
            pos1 = (self.n_vocab + 1 + 3) * self.n_sites + self.t  
            self.state[pos1] = 0
            try:
                self.state[pos1 + 1] = 1
            except:
                pass
            # Updating OHE of element type based on action
            pos2 = (self.n_vocab + 1 + 3) * self.t 
            self.state[pos2 : pos2 + self.n_vocab + 1][action] = 1
            self.state[pos2 : pos2 + self.n_vocab + 1][-1] = 0
            reward = 0
            self.t += 1
#             print(self.species_ind[action], end = ', ')
            if self.species_ind[action] == self.ele[self.t - 1]:
#                 reward = 1
                self.ret += 1
            else:
                reward = 0
#             reward = 0
        else:
            done = True
            # reward = self.calc_reward(self.calc_energy(self.state))
#             reward = self.proxy_reward(self.state)
            reward = 0
            reward = self.ret
#             print()
#             print(reward)


        info = {}
        # print(self.state)
        return (
            self.state, 
            reward,
            done,
            None,
            info
        )
    def seed(self, seed = 0):
        """
        random seed
        """
        self._seed = seed
    def close(self):
        pass





class CrystalEnvV3(BaseEnv):
    def __init__(self, n_vocab = 4, n_sites = 48, species_ind = {0:'Cu', 1:'P', 2:'N', 3:'O'},  #{0:'Cu', 1:'P', 2:'N', 3:'O'}, 
                 atom_num_dict = {}, file_name = '/network/scratch/p/prashant.govindarajan/crystal_design_project/code/Cu3P2NO6.cif', env_name = 'CrystalEnvV3', seed = 42, **kwargs):
        """
        Crystal structure environment
        n_vocab : vocabulary size (number of elements in the action space)
        n_sites : number of sites in the unit cell
        species_ind : Index for elements
        atom_num_dict : Dictionary of atomic numbers
        file_name : CIF File containing crystal (temporary)
        env_name : name of environment
        seed : random seed value
        """
        
        self.env_name = env_name
        self.n_vocab = n_vocab
        # Size of state vector
        self.state_size = 6 + (5 + n_vocab) * n_sites  ## 9 for lattice, 1 + n_vocab for each element and blank space, 3 for coordinates, n_sites for tracking position
        self.species_ind = species_ind
        self.atom_num_dict = atom_num_dict
        self.n_sites = n_sites
        self.t = 0
        self._seed = seed
        self.file_name = file_name

        #Action space
        self.action_space = Discrete(self.n_vocab)
        # State space
        self.observation_space = Box(low = np.array([-np.inf] * self.state_size), high = np.array([np.inf] * self.state_size))

        self._env_spec = self.create_env_spec(self.env_name, **kwargs)

        ### Temporary ###
        # Parse CIF File
        self.mat =  cif.CifParser(self.file_name).as_dict()['Cu3P2NO6']
        # Get lattice vector
        self.lattice = cif.CifParser(self.file_name).get_lattice(self.mat)
        self.a = self.lattice.a
        self.b = self.lattice.b
        self.c = self.lattice.c
        self.alpha = self.lattice.alpha
        self.beta = self.lattice.beta
        self.gamma = self.lattice.gamma       
        #self.lat_mat = np.ravel(self.lattice.matrix)
        # Initialize state
        self.state = self.random_initial_state()
        # print(self.state)
        
        self.ele = self.mat['_atom_site_type_symbol']
        
        ### Test ###
        self.ret = 0

    def random_initial_state(self):
        """
        Initialize state to default (for now, skeleton of one crystal)
        State --> [lattice, atom positions, coordinates, tracking pointer]
        To do : choose a random crystal skeleton from a collection
        """

        ele = self.mat['_atom_site_type_symbol']

        coords_x = self.mat['_atom_site_fract_x']
        coords_y = self.mat['_atom_site_fract_y']
        coords_z = self.mat['_atom_site_fract_z']

        state = np.array([])

        for i in range(self.n_sites):
            tmp = np.zeros(self.n_vocab + 1)
            tmp[-1] = 1
            c = np.array([float(coords_x[i]), float(coords_y[i]), float(coords_z[i])])
            state = np.concatenate([state, tmp, c])
        pointer = np.zeros(self.n_sites)
        pointer[0] = 1
        lat = np.array([self.a, self.b, self.a, self.alpha, self.beta, self.gamma])
        state = np.concatenate([lat, state, pointer])

        return state

    def calc_energy(self, state):
        """
        Calculate energy of material using SymmetryMode (Pymatgen)
        state : representation of state
        returns : Energy calculated by Symmetry Model
        """
        ele = []
        coords = []
        for i in range(9, self.state_size - self.n_sites, self.n_vocab + 4):
            predictions = state[i : i + self.n_vocab]
            # print(predicitons)
            positions = list(state[i + self.n_vocab + 1 : i + self.n_vocab + 4])
            index = np.where(predictions)[0][0]
            ele.append(self.species_ind[index])
            coords.append(positions)
        struct = S.Structure(lattice = self.lattice, species = ele, coords = coords)
        energy = em.SymmetryModel().get_energy(struct)
        return energy

    def calc_reward(self, energy, thresh = 0):
        """
        Calculate reward using energy
        """
        ### trial : reward = energy for the timebeing
        reward = energy
        ###
        
        return reward

    def proxy_reward(self, state):
        true_ele = self.mat['_atom_site_type_symbol']
        pred_ele = []
        reward = 0
        k = 0
        for i in range(9, self.state_size - self.n_sites, self.n_vocab + 4):
            predictions = state[i : i + self.n_vocab]
#             print(predictions, len(predictions))
            index = np.where(predictions)[0][0]
            reward += int(true_ele[k] == self.species_ind[index])
            k += 1
        return reward

    def create_env_spec(self, env_name, **kwargs):
        """
        Each family of environments have their own type of observations and actions.
        You can add support for more families here by modifying observation_space and action_space.
        """
        return EnvSpec(
            env_name=env_name,
            observation_space=[self.observation_space],
            action_space=[self.action_space],
        )
    def reset(self):
        self.t = 0
        self.ret = 0
        self.state = self.random_initial_state()
        return self.state, None

    def step(self, action):
        """
        step function
        action : action chosen by the agent (element)
        """

        done = False
        if self.t < self.n_sites:
            # Moving pointer by one position
            pos1 = 6 + (self.n_vocab + 1 + 3) * self.n_sites + self.t  
            self.state[pos1] = 0
            try:
                self.state[pos1 + 1] = 1
            except:
                pass
            # Updating OHE of element type based on action
            pos2 = 6 + (self.n_vocab + 1 + 3) * self.t 
            self.state[pos2 : pos2 + self.n_vocab + 1][action] = 1
            self.state[pos2 : pos2 + self.n_vocab + 1][-1] = 0
            reward = 0
            self.t += 1
#             print(self.species_ind[action], end = ', ')
            if self.species_ind[action] == self.ele[self.t - 1]:
#                 reward = 1
                self.ret += 1
            else:
                reward = 0
#             reward = 0
        else:
            done = True
            # reward = self.calc_reward(self.calc_energy(self.state))
#             reward = self.proxy_reward(self.state)
            reward = 0
            reward = self.ret
#             print()
#             print(reward)


        info = {}
        # print(self.state)
        return (
            self.state, 
            reward,
            done,
            None,
            info
        )
    def seed(self, seed = 0):
        """
        random seed
        """
        self._seed = seed
    def close(self):
        pass
    
    
class CrystalEnvCubic(BaseEnv):
    
    def __init__(self, file_name = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/data/materials_project_cubic.pkl', env_name = 'CrystalEnvCubic', seed = 42, **kwargs):
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
        self.data = pd.read_pickle(file_name).reset_index()
        self.num_samples = self.data.shape[0]
        self.n_vocab = 85
        self.action_space = Discrete(self.n_vocab)
        self.NMAX = 336
        self.MAX_SIZE = 6 + (5 + self.n_vocab) * self.NMAX
        self.observation_space = Box(low = np.array([-np.inf] * self.MAX_SIZE), high = np.array([np.inf] * self.MAX_SIZE))
        self._env_spec = self.create_env_spec(self.env_name, **kwargs)
        self.init_vocab()
        self.state = self.random_initial_state()
    # WARNING: Decompyle incomplete
    def init_vocab(self):
        elements = [
            'Cu',
            'F',
            'Eu',
            'Co',
            'C',
            'Xe',
            'Cr',
            'Cl',
            'Zr',
            'Sm',
            'Os',
            'Yb',
            'Nd',
            'H',
            'Be',
            'Fe',
            'Na',
            'Tm',
            'Y',
            'Ag',
            'N',
            'Ni',
            'K',
            'In',
            'Lu',
            'Sr',
            'Ir',
            'Rh',
            'S',
            'Dy',
            'Tl',
            'Ge',
            'Tc',
            'Nb',
            'Au',
            'As',
            'Tb',
            'Se',
            'Cd',
            'Bi',
            'O',
            'Np',
            'Pa',
            'Pr',
            'Pd',
            'Pt',
            'Re',
            'Cs',
            'Ta',
            'Hf',
            'B',
            'Er',
            'Ga',
            'Ce',
            'Ti',
            'Zn',
            'Sb',
            'Mg',
            'Ca',
            'Te',
            'Al',
            'W',
            'Sc',
            'Rb',
            'Sn',
            'Mn',
            'I',
            'Ru',
            'Th',
            'Pu',
            'Si',
            'V',
            'La',
            'Ho',
            'Br',
            'Ac',
            'Ba',
            'Gd',
            'Pb',
            'U',
            'Hg',
            'Li',
            'Mo',
            'P',
            'Pm']
        self.n_vocab = len(elements)
        self.species_ind = {i:elements[i] for i in range(self.n_vocab)}
    
    def random_initial_state(self):
        '''
        Initialize state to default (for now, skeleton of one crystal)
        State --> [lattice, atom positions, coordinates, tracking pointer]
        To do : choose a random crystal skeleton from a collection
        '''
        self.t = 0
        self.ret = 0
        ind = np.random.choice(range(self.num_samples))
        cif_string = self.data.loc[ind]['cif']
        mat_tmp = cif.CifParser.from_string(cif_string)
        mat = mat_tmp.as_dict()
        self.mat = mat[list(mat.keys())[0]]
        self.lattice = cif.CifParser.from_string(cif_string).get_lattice(self.mat)
        self.a = self.lattice.a
        self.b = self.lattice.b
        self.c = self.lattice.c
        self.alpha = self.lattice.alpha
        self.beta = self.lattice.beta
        self.gamma = self.lattice.gamma
        self.ele = self.mat['_atom_site_type_symbol']
        self.n_sites = len(self.ele)
        self.state_size = 6 + (5 + self.n_vocab) * self.n_sites
        coords_x = self.mat['_atom_site_fract_x']
        coords_y = self.mat['_atom_site_fract_y']
        coords_z = self.mat['_atom_site_fract_z']
        state = np.array([])
        for i in range(self.n_sites):
            tmp = np.zeros(self.n_vocab + 1)
            tmp[-1] = 1
            c = np.array([
                float(coords_x[i]),
                float(coords_y[i]),
                float(coords_z[i])])
            state = np.concatenate([
                state,
                tmp,
                c])
        pointer = np.zeros(self.n_sites)
        pointer[0] = 1
        lat = np.array([
            self.a,
            self.b,
            self.a,
            self.alpha,
            self.beta,
            self.gamma])
        state = np.concatenate([
            lat,
            state,
            pointer])
        state = np.concatenate([
            state,
            np.zeros(self.MAX_SIZE - self.state_size)])
        return state

    
    def calc_energy(self, state):
        '''
        Calculate energy of material using SymmetryMode (Pymatgen)
        state : representation of state
        returns : Energy calculated by Symmetry Model
        '''
        ele = []
        coords = []
        for i in range(9, self.state_size - self.n_sites, self.n_vocab + 4):
            predictions = state[i:i + self.n_vocab]
            positions = list(state[i + self.n_vocab + 1:i + self.n_vocab + 4])
            index = np.where(predictions)[0][0]
            ele.append(self.species_ind[index])
            coords.append(positions)
        struct = S.Structure(self.lattice, ele, coords, **('lattice', 'species', 'coords'))
        energy = em.SymmetryModel().get_energy(struct)
        return energy

    
    def calc_reward(self, energy, thresh = (0,)):
        '''
        Calculate reward using energy
        '''
        reward = energy
        return reward

    
    def proxy_reward(self, state):
        true_ele = self.mat['_atom_site_type_symbol']
        pred_ele = []
        reward = 0
        k = 0
        for i in range(9, self.state_size - self.n_sites, self.n_vocab + 4):
            predictions = state[i:i + self.n_vocab]
            index = np.where(predictions)[0][0]
            reward += int(true_ele[k] == self.species_ind[index])
            k += 1
        return reward

    
    def create_env_spec(self, env_name, **kwargs):
        """
        Each family of environments have their own type of observations and actions.
        You can add support for more families here by modifying observation_space and action_space.
        """
        return EnvSpec(
            env_name=env_name,
            observation_space=[self.observation_space],
            action_space=[self.action_space],
        )

    
    def reset(self):
        self.t = 0
        self.ret = 0
        self.state = self.random_initial_state()
        return (self.state, None)

    
    def step(self, action):
        '''
        step function
        action : action chosen by the agent (element)
        '''
        done = False
        if self.t < self.n_sites:
            pos1 = 6 + (self.n_vocab + 1 + 3) * self.n_sites + self.t
            self.state[pos1] = 0
            
            try:
                self.state[pos1 + 1] = 1
            finally:
                pass
            pos2 = 6 + (self.n_vocab + 1 + 3) * self.t
            self.state[pos2:pos2 + self.n_vocab + 1][action] = 1
            self.state[pos2:pos2 + self.n_vocab + 1][-1] = 0
            reward = 0
            self.t += 1
            if self.species_ind[action] == self.ele[self.t - 1]:
                self.ret += 1
            else:
                reward = 0
        
        else:
            done = True
            reward = 0
            reward = self.ret / self.n_sites
        info = {}
        return (self.state, reward, done, None, info)
    
    def seed(self, seed = (0,)):
        """
        random seed
        """        
        self._seed = seed

    def close(self):
        pass




