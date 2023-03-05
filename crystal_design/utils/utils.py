import torch
import dgl
from copy import deepcopy

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_function(graph_batches, n_vocab = 56):
    batch_size = len(graph_batches)

    collate_atomic_number = torch.zeros((batch_size * 5, n_vocab + 1))
    collate_positions = torch.zeros((batch_size * 5, 3))
    collate_lengths = torch.zeros((batch_size, 11))
    collate_actions = torch.zeros((batch_size,))

    for i in range(batch_size):
        crystal = graph_batches[i]#.to(device = 'cuda:0')
        edges_u, edges_v = deepcopy(crystal.edges())
        edges_u += 5 * i
        edges_v += 5 * i
        # g.add_edges(edges[0] + 5 * i, edges[1] + 5 * i)
        atomic_number = crystal.ndata['atomic_number'] ## 5 X 57
        positions = crystal.ndata['position'] ## 5 x 3
        # true_atomic_number = crystal.ndata['true_atomic_number'] ## 5 x 57
        lengths_angles_focus = crystal.lengths_angles_focus #.unsqueeze(0) ## 11
        action = crystal.action #.unsqueeze(0)
        if i == 0:
            collate_edges_u = edges_u.unsqueeze(1)
            collate_edges_v = edges_v.unsqueeze(1)
        else:
            collate_edges_u = torch.cat([collate_edges_u, edges_u.unsqueeze(1)], dim = 0)
            collate_edges_v = torch.cat([collate_edges_v, edges_v.unsqueeze(1)], dim = 0)
        collate_atomic_number[5*i : (5*i + 5), :] = atomic_number
        collate_positions[5*i : (5*i + 5), :] = positions
        collate_lengths[i, :] = lengths_angles_focus
        collate_actions[i] = action

    ## Create graph
    g = dgl.graph(data = (collate_edges_u.squeeze(1), collate_edges_v.squeeze(1)), num_nodes = batch_size * 5).to(device='cpu')
    g.ndata['atomic_number'] = collate_atomic_number
    g.ndata['position'] = collate_positions
    g.action = collate_actions.long().cuda() #.squeeze(1)#.cuda().squeeze(1)
    g.lengths_angles_focus = collate_lengths.cuda()

    return g.to(device = 'cuda:0')

def collate_functionV2(batch_list):
    batch_size = len(batch_list)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    action_list = [None] * batch_size
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    for i in range(batch_size):
        data_dict = batch_list[i]
        atomic_number_list[i] = data_dict['atomic_number']
        # true_atomic_number_list[i] = data_dict['true_atomic_number']
        position_list[i] = data_dict['position']
        action_list[i] = data_dict['action']
        laf_list[i] = data_dict['laf']
        edges_u[i] =  data_dict['edges_u'] + 5 * i
        edges_v[i] =  data_dict['edges_v'] + 5 * i
    g = dgl.graph(data = (torch.cat(edges_u), torch.cat(edges_v)), num_nodes = batch_size * 5)#.to(device='cpu')
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    # g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.action = torch.stack(action_list).squeeze(1)#.cuda()
    g.lengths_angles_focus = torch.stack(laf_list)#.cuda()
    return g            

def collate_functionV3(batch_list):
    batch_size = len(batch_list)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    action_list = [None] * batch_size
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    for i in range(batch_size):
        data_dict = batch_list[i]
        atomic_number_list[i] = data_dict['atomic_number']
        # true_atomic_number_list[i] = data_dict['true_atomic_number']
        position_list[i] = data_dict['position']
        action_list[i] = data_dict['action']
        laf_list[i] = data_dict['laf']
        edges_u[i] =  data_dict['edges_u'] + 5 * i
        edges_v[i] =  data_dict['edges_v'] + 5 * i
        to_jimages[i] = data_dict['to_jimages']
    g = dgl.graph(data = (torch.cat(edges_u), torch.cat(edges_v)), num_nodes = batch_size * 5)#.to(device='cpu')
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    # g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.action = torch.stack(action_list)#.squeeze(1)#.cuda()
    g.lengths_angles_focus = torch.stack(laf_list)#.cuda()
    g.edata['to_jimages'] = torch.cat(to_jimages, dim = 0)
    return g

# def to_cdvae_like(data_path):
#     data = torch.load(data_path)
#     n = len(data)
#     num_atoms_list = []
#     lengths_list = []
#     angles_list = []
#     atom_types_list = []
#     for batch in data:
#         num_atoms_list.append(5) #= data.num_atoms
#         lengths_list.append(batch.lengths_angles_focus[:, :3])
#         angles_list.append(batch.lengths_angles_focus[:,3:6])
#         atom_types_list.append(batch.ndata['atomic_number'])








# import torch
# import numpy as np
# from pymatgen.core.structure import Structure
# from pymatgen.analysis import energy_models as em
# import pymatgen.io.cif as cif
# from tqdm import tqdm 
# from pymatgen.io.cif import CifWriter
# from pymatgen.core.lattice import Lattice

# if __name__ == '__main__':
#     data = torch.load('/home/mila/p/prashant.govindarajan/scratch/COMP760-Project/cdvae/cosine/cos_eval_gen_new.pt') ## Change path
#     N = data['num_atoms'].shape[1]
#     j = 0
#     for i in tqdm(range(N)):
#         num_atoms = int(data['num_atoms'][0][i].cpu().numpy())
#         lengths = tuple(data['lengths'][0][i].cpu().numpy())
#         angles = tuple(data['angles'][0][i].cpu().numpy())
#         lattice_params = lengths + angles
#         atom_types = list(data['atom_types'][0][j:j+num_atoms].cpu().numpy())
#         frac_coords = data['frac_coords'][0][j:j+num_atoms,:].cpu().numpy()
#         j += num_atoms
#         canonical_crystal = Structure(lattice = Lattice.from_parameters(*lattice_params),
#                                     species = atom_types, coords = frac_coords, coords_are_cartesian = False)
#         writer = CifWriter(canonical_crystal)
#         writer.write_file('/home/mila/p/prashant.govindarajan/scratch/COMP760-Project/cdvae/cosine/generated_cifs_new/'+str(i)+'.cif')   ##Change path


    

    




