import torch
from pymatgen.core.structure import Structure
from tqdm import tqdm 
from pymatgen.io.cif import CifWriter
from pymatgen.core.lattice import Lattice
import mendeleev
import os

ELEMENTS = ['Cs', 'Er', 'Xe', 'Tc', 'Eu', 'Gd', 'Li', 'Hf', 'Dy', 'F', 'Te', 'Ti', 'Hg', 'Bi', 'Pr', 'Ne', 'Sm', 'Be', 'Au', 'Pb', 'C', 'Zr', 'Ir', 'Pd', 'Sc', 'Yb', 'Os', 'Nb', 'Ac', 'Rb', 'Al', 'P', 'Ga', 'Na', 'Cr', 'Ta', 'Br', 'Pu', 'Ge', 'Tb', 'La', 'Se', 'V', 'Pa', 'Ni', 'In', 'Cu', 'Fe', 'Co', 'Pm', 'N', 'K', 'Ca', 'Rh', 'B', 'Tm', 'I', 'Ho', 'Sb', 'As', 'Tl', 'Ru', 'U', 'Np', 'Cl', 'Re', 'Ag', 'Ba', 'H', 'O', 'Mg', 'W', 'Sn', 'Mo', 'Pt', 'Zn', 'Sr', 'S', 'Kr', 'Cd', 'Si', 'Y', 'Lu', 'Th', 'Nd', 'Mn', 'He', 'Ce']
SPECIES_IND = {i:mendeleev.element(ELEMENTS[i]).atomic_number for i in range(len(ELEMENTS))}
SPECIES_IND_INV = {mendeleev.element(ELEMENTS[i]).atomic_number:i for i in range(len(ELEMENTS))}

def to_cif(data_path, save_path):
    data = torch.load(data_path) 
    try:
        os.mkdir(save_path)
    except:
        pass

    N = len(data)
    j = 0

    for i in tqdm(range(N)):
        num_atoms = int(data[i].ndata['atomic_number'].shape[0])
        laf = data[i].lengths_angles_focus[0]
        lengths = laf[:3].tolist()
        angles = laf[3:6].tolist()
        lattice_params = lengths + angles
        atomic_number = data[i].ndata['atomic_number'][:,:-2]
        atomic_number = torch.argmax(atomic_number, dim = 1)
        ind = data[i].inds
        atom_types = [SPECIES_IND[int(atomic_number[i].cpu().numpy())] for i in range(atomic_number.shape[0])]
        coords = data[i].ndata['position'].cpu().numpy()
        j += num_atoms
        canonical_crystal = Structure(lattice = Lattice.from_parameters(*lattice_params),
                                    species = atom_types, coords = coords, coords_are_cartesian = True)
        writer = CifWriter(canonical_crystal)
        writer.write_file(save_path+'/'+str(ind[0].numpy())+'.cif')  

