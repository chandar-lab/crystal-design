import torch
import dgl
from copy import deepcopy
import numpy as np
from p_tqdm import p_map, p_umap
from matgl.layers import MLP, BondExpansion
from crystal_design.utils.compute_prop import Crystal, OptEval, GenEval
# from cdvae.common.data_utils import cart_to_frac_coords
from torch.nn.functional import one_hot


SI_BG = 1.12
NUM_WORKERS = 16
print(SI_BG)
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

def create_feature_lists(data_dict, data_dict_next, action, reward, bandgap, done, sum_n_atoms):
    atomic_number = data_dict['atomic_number']
    atomic_number_next = data_dict_next['atomic_number']
    n_atoms = data_dict['atomic_number'].shape[0]
    true_atomic_number = data_dict['true_atomic_number']
    position = data_dict['coordinates']
    laf = torch.cat((data_dict['laf'], torch.tensor([4.0]))) #data_dict['laf'] #torch.cat((data_dict['laf'], torch.zeros(494-n_atoms)))
    edges_u =  data_dict['edges'][0] + sum_n_atoms
    edges_v =  data_dict['edges'][1] + sum_n_atoms
    to_jimages = data_dict['etype']

    return atomic_number, atomic_number_next, true_atomic_number, position, laf, edges_u, edges_v, to_jimages, n_atoms, action, reward, bandgap, done


def collate_function_megnet(states, actions, rewards, bandgaps, next_states, dones):
    batch_size = len(states)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    sum_n_atoms = 0
    n_atoms_list = []
    n_edges_list = []
    ind_list = []
    atomic_number_list_next = [None] * batch_size

    for i in range(batch_size):
        data_dict = states[i]
        atomic_number_list[i] = data_dict['atomic_number']
        n_atoms = data_dict['atomic_number'].shape[0]
        true_atomic_number_list[i] = data_dict['true_atomic_number']
        position_list[i] = data_dict['coordinates']
        laf_list[i] = torch.cat((data_dict['laf'], torch.tensor([SI_BG]))) #data_dict['laf'] #torch.cat((data_dict['laf'], torch.zeros(494-n_atoms)))
        edges_u_single =  data_dict['edges'][0] + sum_n_atoms
        edges_v_single =  data_dict['edges'][1] + sum_n_atoms
        edges_cat = torch.cat([edges_u_single[:,None], edges_v_single[:,None]], dim = 1)
        to_jimages[i] = data_dict['etype']
        ind = torch.where(torch.prod(to_jimages[i]==0, dim = 1))[0]
        edges_cat = edges_cat[ind, :]
        edges_cat = torch.unique(edges_cat, dim = 0)
        edges_u[i] = edges_cat[:,0]
        edges_v[i] = edges_cat[:,1]
        n_edges_list.append(edges_cat.shape[0])
        sum_n_atoms += n_atoms
        n_atoms_list.append(n_atoms)
        # ind_list.append(data_dict['ind'])

        data_dict_next = next_states[i]
        atomic_number_list_next[i] = data_dict_next['atomic_number']
    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    # to_jimages = torch.cat(to_jimages, dim = 0)
    # ind = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat = edges_cat[ind, :]
    # edges_cat = torch.unique(edges_cat, dim = 0)
    position = torch.cat(position_list, dim = 0)
    edata = torch.norm(position[edges_cat[:,0]] - position[edges_cat[:,1]], dim = 1)
    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    # g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.lengths_angles_focus = torch.stack(laf_list)#.cuda()
    g.edata['e_feat'] = edata
    # g.edata['to_jimages'] = torch.cat(to_jimages, dim = 0) #torch.cat(to_jimages, dim = 0)
    g.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g.set_batch_num_edges(torch.tensor(n_edges_list))
    # g.inds = torch.tensor(ind_list)

    # edges_cat_next = torch.cat([torch.cat(edges_u_next)[:,None], torch.cat(edges_v_next)[:,None]], dim = 1)
    # to_jimages_next = torch.cat(to_jimages_next, dim = 0)
    # ind_next = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat_next = edges_cat_next[ind_next, :]
    # edges_cat_next = torch.unique(edges_cat_next, dim = 0)

    g_next = dgl.graph(data = torch.unbind((edges_cat), dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list, dim = 0)
    g_next.lengths_angles_focus = torch.stack(laf_list)#.cuda()
    g_next.edata['e_feat'] = edata
    # g_next.edata['to_jimages'] = to_jimages_next #torch.cat(to_jimages_next, dim = 0)
    # g_next.n_atoms = torch.tensor(n_atoms_list)
    g_next.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g_next.set_batch_num_edges(torch.tensor(n_edges_list))
    # g_next.inds = torch.tensor(ind_list)

    return g, actions, g_next, torch.tensor(rewards), torch.tensor(bandgaps), torch.tensor(dones)

def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)

def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index[:,0], edge_index[:,1]

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattice, torch.tensor(num_bonds), dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out

def collate_function_megnet_multigraphs(states, actions, rewards, bandgaps, next_states, dones):
    batch_size = len(states)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    sum_n_atoms = 0
    n_atoms_list = []
    n_edges_list = []
    ind_list = []
    atomic_number_list_next = [None] * batch_size
    num_edges = []

    for i in range(batch_size):
        data_dict = states[i]
        atomic_number_list[i] = data_dict['atomic_number']
        n_atoms = data_dict['atomic_number'].shape[0]
        true_atomic_number_list[i] = data_dict['true_atomic_number']
        position_list[i] = data_dict['coordinates']
        laf_list[i] = torch.cat((data_dict['laf'], torch.tensor([SI_BG]))) #data_dict['laf'] #torch.cat((data_dict['laf'], torch.zeros(494-n_atoms)))
        edges_u_single =  data_dict['edges'][0] + sum_n_atoms
        edges_v_single =  data_dict['edges'][1] + sum_n_atoms
        edges_cat = torch.cat([edges_u_single[:,None], edges_v_single[:,None]], dim = 1)
        num_edges.append(edges_cat.shape[0])
        to_jimages[i] = data_dict['etype']
        # ind = torch.where(torch.prod(to_jimages[i]==0, dim = 1))[0]
        # edges_cat = edges_cat[ind, :]
        # edges_cat = torch.unique(edges_cat, dim = 0)
        edges_u[i] = edges_cat[:,0]
        edges_v[i] = edges_cat[:,1]
        n_edges_list.append(edges_cat.shape[0])
        sum_n_atoms += n_atoms
        n_atoms_list.append(n_atoms)
        # ind_list.append(data_dict['ind'])

        data_dict_next = next_states[i]
        atomic_number_list_next[i] = data_dict_next['atomic_number']
    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    to_jimages = torch.cat(to_jimages, dim = 0)
    # ind = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat = edges_cat[ind, :]
    # edges_cat = torch.unique(edges_cat, dim = 0)
    position = torch.cat(position_list, dim = 0)
    laf_list = torch.stack(laf_list)
    out = get_pbc_distances(position, edges_cat, lengths = laf_list[:,:3], angles = laf_list[:,3:6], to_jimages = to_jimages, 
                            num_atoms = n_atoms_list, num_bonds=num_edges, coord_is_cart=True)
    edata = out['distances'] #torch.norm(position[edges_cat[:,0]] - position[edges_cat[:,1]], dim = 1)
    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    # g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.lengths_angles_focus = laf_list#.cuda()
    g.edata['e_feat'] = edata
    # g.edata['to_jimages'] = torch.cat(to_jimages, dim = 0) #torch.cat(to_jimages, dim = 0)
    g.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g.set_batch_num_edges(torch.tensor(n_edges_list))
    # g.inds = torch.tensor(ind_list)

    # edges_cat_next = torch.cat([torch.cat(edges_u_next)[:,None], torch.cat(edges_v_next)[:,None]], dim = 1)
    # to_jimages_next = torch.cat(to_jimages_next, dim = 0)
    # ind_next = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat_next = edges_cat_next[ind_next, :]
    # edges_cat_next = torch.unique(edges_cat_next, dim = 0)

    g_next = dgl.graph(data = torch.unbind((edges_cat), dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list, dim = 0)
    g_next.lengths_angles_focus = laf_list#.cuda()
    g_next.edata['e_feat'] = edata
    # g_next.edata['to_jimages'] = to_jimages_next #torch.cat(to_jimages_next, dim = 0)
    # g_next.n_atoms = torch.tensor(n_atoms_list)
    g_next.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g_next.set_batch_num_edges(torch.tensor(n_edges_list))
    # g_next.inds = torch.tensor(ind_list)

    return g, actions, g_next, torch.tensor(rewards), torch.tensor(bandgaps), torch.tensor(dones)


def collate_function_megnet_multigraphs_torchRL(batch):
    
    batch_size = len(batch)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    sum_n_atoms = 0
    n_atoms_list = []
    n_edges_list = []
    action_list = []
    ind_list = []
    atomic_number_list_next = [None] * batch_size
    num_edges = []
    reward_list = []
    bandgap_list = []
    dones_list = []

    for i in range(batch_size):
        data_dict = batch[i]
        observation, action, next_observation, reward, bandgap, done = data_dict
        atomic_number_list[i] = observation['atomic_number']
        n_atoms = observation['atomic_number'].shape[0]
        true_atomic_number_list[i] = observation['true_atomic_number']
        position_list[i] = observation['coordinates']
        laf_list[i] = torch.cat((observation['laf'], torch.tensor([SI_BG]))) #data_dict['laf'] #torch.cat((data_dict['laf'], torch.zeros(494-n_atoms)))
        edges_u_single =  observation['edges'][0] + sum_n_atoms
        edges_v_single =  observation['edges'][1] + sum_n_atoms
        edges_cat = torch.cat([edges_u_single[:,None], edges_v_single[:,None]], dim = 1)
        num_edges.append(edges_cat.shape[0])
        to_jimages[i] = observation['etype']
        # ind = torch.where(torch.prod(to_jimages[i]==0, dim = 1))[0]
        # edges_cat = edges_cat[ind, :]
        # edges_cat = torch.unique(edges_cat, dim = 0)
        edges_u[i] = edges_cat[:,0]
        edges_v[i] = edges_cat[:,1]
        n_edges_list.append(edges_cat.shape[0])
        sum_n_atoms += n_atoms
        n_atoms_list.append(n_atoms)
        # ind_list.append(data_dict['ind'])

        action_list.append(action)
        reward_list.append(reward)
        bandgap_list.append(bandgap)
        dones_list.append(done)

        atomic_number_list_next[i] = next_observation['atomic_number']
    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    to_jimages = torch.cat(to_jimages, dim = 0)
    # ind = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat = edges_cat[ind, :]
    # edges_cat = torch.unique(edges_cat, dim = 0)
    position = torch.cat(position_list, dim = 0)
    laf_list = torch.stack(laf_list)
    out = get_pbc_distances(position, edges_cat, lengths = laf_list[:,:3], angles = laf_list[:,3:6], to_jimages = to_jimages, 
                            num_atoms = n_atoms_list, num_bonds=num_edges, coord_is_cart=True)
    edata = out['distances'] #torch.norm(position[edges_cat[:,0]] - position[edges_cat[:,1]], dim = 1)
    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    # g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.lengths_angles_focus = laf_list#.cuda()
    g.edata['e_feat'] = edata
    # g.edata['to_jimages'] = torch.cat(to_jimages, dim = 0) #torch.cat(to_jimages, dim = 0)
    g.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g.set_batch_num_edges(torch.tensor(n_edges_list))
    # g.inds = torch.tensor(ind_list)

    # edges_cat_next = torch.cat([torch.cat(edges_u_next)[:,None], torch.cat(edges_v_next)[:,None]], dim = 1)
    # to_jimages_next = torch.cat(to_jimages_next, dim = 0)
    # ind_next = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat_next = edges_cat_next[ind_next, :]
    # edges_cat_next = torch.unique(edges_cat_next, dim = 0)

    g_next = dgl.graph(data = torch.unbind((edges_cat), dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list, dim = 0)
    g_next.lengths_angles_focus = laf_list#.cuda()
    g_next.edata['e_feat'] = edata
    # g_next.edata['to_jimages'] = to_jimages_next #torch.cat(to_jimages_next, dim = 0)
    # g_next.n_atoms = torch.tensor(n_atoms_list)
    g_next.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g_next.set_batch_num_edges(torch.tensor(n_edges_list))
    # g_next.inds = torch.tensor(ind_list)

    return g, torch.tensor(action_list), g_next, torch.tensor(reward_list), torch.tensor(bandgap_list), torch.tensor(dones_list)


def collate_function_megnet_multigraphs_eval(batch):
    
    batch_size = len(batch)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    sum_n_atoms = 0
    n_atoms_list = []
    n_edges_list = []
    action_list = []
    ind_list = []
    atomic_number_list_next = [None] * batch_size
    num_edges = []
    reward_list = []
    bandgap_list = []
    dones_list = []

    for i in range(batch_size):
        data_dict = batch[i]
        observation, action, next_observation, reward, done = data_dict
        atomic_number_list[i] = observation['atomic_number']
        n_atoms = observation['atomic_number'].shape[0]
        true_atomic_number_list[i] = observation['true_atomic_number']
        position_list[i] = observation['coordinates']
        laf_list[i] = torch.cat((observation['laf'], torch.tensor([SI_BG]))) #data_dict['laf'] #torch.cat((data_dict['laf'], torch.zeros(494-n_atoms)))
        edges_u_single =  observation['edges'][0] + sum_n_atoms
        edges_v_single =  observation['edges'][1] + sum_n_atoms
        edges_cat = torch.cat([edges_u_single[:,None], edges_v_single[:,None]], dim = 1)
        num_edges.append(edges_cat.shape[0])
        to_jimages[i] = observation['etype']
        # ind = torch.where(torch.prod(to_jimages[i]==0, dim = 1))[0]
        # edges_cat = edges_cat[ind, :]
        # edges_cat = torch.unique(edges_cat, dim = 0)
        edges_u[i] = edges_cat[:,0]
        edges_v[i] = edges_cat[:,1]
        n_edges_list.append(edges_cat.shape[0])
        sum_n_atoms += n_atoms
        n_atoms_list.append(n_atoms)
        ind_list.append(observation['ind'])

        action_list.append(action)
        reward_list.append(reward)
        dones_list.append(done)

        atomic_number_list_next[i] = next_observation['atomic_number']
    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    to_jimages = torch.cat(to_jimages, dim = 0)
    # ind = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat = edges_cat[ind, :]
    # edges_cat = torch.unique(edges_cat, dim = 0)
    position = torch.cat(position_list, dim = 0)
    laf_list = torch.stack(laf_list)
    out = get_pbc_distances(position, edges_cat, lengths = laf_list[:,:3], angles = laf_list[:,3:6], to_jimages = to_jimages, 
                            num_atoms = n_atoms_list, num_bonds=num_edges, coord_is_cart=True)
    edata = out['distances'] #torch.norm(position[edges_cat[:,0]] - position[edges_cat[:,1]], dim = 1)
    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.lengths_angles_focus = laf_list#.cuda()
    g.edata['e_feat'] = edata
    # g.edata['to_jimages'] = torch.cat(to_jimages, dim = 0) #torch.cat(to_jimages, dim = 0)
    g.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g.set_batch_num_edges(torch.tensor(n_edges_list))
    g.inds = torch.tensor(ind_list)

    # edges_cat_next = torch.cat([torch.cat(edges_u_next)[:,None], torch.cat(edges_v_next)[:,None]], dim = 1)
    # to_jimages_next = torch.cat(to_jimages_next, dim = 0)
    # ind_next = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat_next = edges_cat_next[ind_next, :]
    # edges_cat_next = torch.unique(edges_cat_next, dim = 0)

    g_next = dgl.graph(data = torch.unbind((edges_cat), dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list, dim = 0)
    g_next.lengths_angles_focus = laf_list#.cuda()
    g_next.edata['e_feat'] = edata
    # g_next.edata['to_jimages'] = to_jimages_next #torch.cat(to_jimages_next, dim = 0)
    # g_next.n_atoms = torch.tensor(n_atoms_list)
    g_next.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g_next.set_batch_num_edges(torch.tensor(n_edges_list))
    g_next.inds = torch.tensor(ind_list)

    return g#, torch.tensor(action_list), g_next, torch.tensor(reward_list), torch.tensor(bandgap_list), torch.tensor(dones_list)

def collate_function_megnet_eval(batch):
    batch_size = len(batch)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    sum_n_atoms = 0
    n_atoms_list = []
    n_edges_list = []
    ind_list = []
    atomic_number_list_next = [None] * batch_size

    for i in range(batch_size):
        data_dict = batch[i][0]
        atomic_number_list[i] = data_dict['atomic_number']
        n_atoms = data_dict['atomic_number'].shape[0]
        true_atomic_number_list[i] = data_dict['true_atomic_number']
        position_list[i] = data_dict['coordinates']
        laf_list[i] = torch.cat((data_dict['laf'], torch.tensor([SI_BG]))) #data_dict['laf'] #torch.cat((data_dict['laf'], torch.zeros(494-n_atoms)))
        edges_u_single =  data_dict['edges'][0] + sum_n_atoms
        edges_v_single =  data_dict['edges'][1] + sum_n_atoms
        edges_cat = torch.cat([edges_u_single[:,None], edges_v_single[:,None]], dim = 1)
        to_jimages[i] = data_dict['etype']
        ind = torch.where(torch.prod(to_jimages[i]==0, dim = 1))[0]
        edges_cat = edges_cat[ind, :]
        edges_cat = torch.unique(edges_cat, dim = 0)
        edges_u[i] = edges_cat[:,0]
        edges_v[i] = edges_cat[:,1]
        n_edges_list.append(edges_cat.shape[0])
        sum_n_atoms += n_atoms
        n_atoms_list.append(n_atoms)
        ind_list.append(data_dict['ind'])

        data_dict_next = data_dict_next = batch[i][2]
        atomic_number_list_next[i] = data_dict_next['atomic_number']
    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    # to_jimages = torch.cat(to_jimages, dim = 0)
    # ind = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat = edges_cat[ind, :]
    # edges_cat = torch.unique(edges_cat, dim = 0)
    position = torch.cat(position_list, dim = 0)
    edata = torch.norm(position[edges_cat[:,0]] - position[edges_cat[:,1]], dim = 1)
    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.lengths_angles_focus = torch.stack(laf_list)#.cuda()
    g.edata['e_feat'] = edata
    # g.edata['to_jimages'] = torch.cat(to_jimages, dim = 0) #torch.cat(to_jimages, dim = 0)
    g.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g.set_batch_num_edges(torch.tensor(n_edges_list))
    g.inds = torch.tensor(ind_list)

    # edges_cat_next = torch.cat([torch.cat(edges_u_next)[:,None], torch.cat(edges_v_next)[:,None]], dim = 1)
    # to_jimages_next = torch.cat(to_jimages_next, dim = 0)
    # ind_next = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat_next = edges_cat_next[ind_next, :]
    # edges_cat_next = torch.unique(edges_cat_next, dim = 0)

    g_next = dgl.graph(data = torch.unbind((edges_cat), dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list, dim = 0)
    g_next.lengths_angles_focus = torch.stack(laf_list)#.cuda()
    g_next.edata['e_feat'] = edata
    # g_next.edata['to_jimages'] = to_jimages_next #torch.cat(to_jimages_next, dim = 0)
    # g_next.n_atoms = torch.tensor(n_atoms_list)
    g_next.set_batch_num_nodes(torch.tensor(n_atoms_list))
    g_next.set_batch_num_edges(torch.tensor(n_edges_list))
    g_next.inds = torch.tensor(ind_list)

    return g#, actions, g_next, torch.tensor(rewards), torch.tensor(bandgaps), torch.tensor(dones)


def collate_function_offline(states, actions, rewards, bandgaps, next_states, dones):
    batch_size = len(states)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    sum_n_atoms = 0
    n_atoms_list = []

    atomic_number_list_next = [None] * batch_size
    # true_atomic_number_list_next = [None] * batch_size
    # position_list_next = [None] * batch_size 
    # laf_list_next = [None] * batch_size
    # edges_u_next = [None] * batch_size
    # edges_v_next = [None] * batch_size
    # to_jimages_next = [None] * batch_size
    # sum_n_atoms_next = 0
    # n_atoms_list_next = []
    
    # batch_size = len(states)
    # n_atoms = [data_dict['atomic_number'].shape[0] for data_dict in states]
    # sum_n_atoms = [0] * batch_size
    # tmp = np.cumsum(n_atoms)
    # sum_n_atoms[1:] = tmp[:-1]

    # (atomic_number_list, 
    # atomic_number_list_next, 
    # true_atomic_number_list, 
    # position_list, 
    # laf_list, 
    # edges_u, 
    # edges_v,
    # to_jimages,
    # n_atoms,
    # actions,
    # rewards,
    # bandgaps,
    # dones) = zip(*p_umap(
    #                     create_feature_lists,
    #                     states,
    #                     next_states,
    #                     actions,
    #                     rewards,
    #                     bandgaps,
    #                     dones,
    #                     sum_n_atoms,
    #                     num_cpus = 32,
    #                     disable=True
    #                 ))
        
    for i in range(batch_size):
        data_dict = states[i]
        atomic_number_list[i] = data_dict['atomic_number']
        n_atoms = data_dict['atomic_number'].shape[0]
        true_atomic_number_list[i] = data_dict['true_atomic_number']
        position_list[i] = data_dict['coordinates']
        laf_list[i] = torch.cat((data_dict['laf'], torch.tensor([SI_BG]))) #data_dict['laf'] #torch.cat((data_dict['laf'], torch.zeros(494-n_atoms)))
        edges_u[i] =  data_dict['edges'][0] + sum_n_atoms
        edges_v[i] =  data_dict['edges'][1] + sum_n_atoms
        sum_n_atoms += n_atoms
        to_jimages[i] = data_dict['etype']
        n_atoms_list.append(n_atoms)

        data_dict_next = next_states[i]
        atomic_number_list_next[i] = data_dict_next['atomic_number']
        n_atoms_next = data_dict_next['atomic_number'].shape[0]
    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    to_jimages = torch.cat(to_jimages, dim = 0)
    ind = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    edges_cat = edges_cat[ind, :]
    edges_cat = torch.unique(edges_cat, dim = 0)
    
    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.lengths_angles_focus = torch.stack(laf_list)#.cuda()
    # g.edata['to_jimages'] = torch.cat(to_jimages, dim = 0) #torch.cat(to_jimages, dim = 0)
    g.n_atoms = torch.tensor(n_atoms_list)


    # edges_cat_next = torch.cat([torch.cat(edges_u_next)[:,None], torch.cat(edges_v_next)[:,None]], dim = 1)
    # to_jimages_next = torch.cat(to_jimages_next, dim = 0)
    # ind_next = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    # edges_cat_next = edges_cat_next[ind_next, :]
    # edges_cat_next = torch.unique(edges_cat_next, dim = 0)

    g_next = dgl.graph(data = torch.unbind((edges_cat), dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list, dim = 0)
    g_next.lengths_angles_focus = torch.stack(laf_list)#.cuda()
    # g_next.edata['to_jimages'] = to_jimages_next #torch.cat(to_jimages_next, dim = 0)
    g_next.n_atoms = torch.tensor(n_atoms_list)

    return g, actions, g_next, torch.tensor(rewards), torch.tensor(bandgaps), torch.tensor(dones)

def collate_function_offline_eval(batch):
    batch_size = len(batch)
    atomic_number_list = [None] * batch_size
    true_atomic_number_list = [None] * batch_size
    position_list = [None] * batch_size 
    laf_list = [None] * batch_size
    edges_u = [None] * batch_size
    edges_v = [None] * batch_size
    to_jimages = [None] * batch_size
    sum_n_atoms = 0
    n_atoms_list = []
    ind_list = []

    atomic_number_list_next = [None] * batch_size
    true_atomic_number_list_next = [None] * batch_size
    position_list_next = [None] * batch_size 
    laf_list_next = [None] * batch_size
    edges_u_next = [None] * batch_size
    edges_v_next = [None] * batch_size
    to_jimages_next = [None] * batch_size
    sum_n_atoms_next = 0
    n_atoms_list_next = []
    MAX = 500
    for i in range(batch_size):
        data_dict = batch[i][0]
        atomic_number_list[i] = data_dict['atomic_number']
        n_atoms = data_dict['atomic_number'].shape[0]
        true_atomic_number_list[i] = data_dict['true_atomic_number']
        position_list[i] = data_dict['coordinates']
        laf_list[i] = torch.cat((data_dict['laf'], torch.tensor([SI_BG]))) #data_dict['laf'] #torch.cat((data_dict['laf'], torch.zeros(494-n_atoms)))
        edges_u[i] =  data_dict['edges'][0] + sum_n_atoms
        edges_v[i] =  data_dict['edges'][1] + sum_n_atoms
        sum_n_atoms += n_atoms
        to_jimages[i] = data_dict['etype']
        n_atoms_list.append(n_atoms)
        ind_list.append(data_dict['ind'])

        data_dict_next = batch[i][2]
        atomic_number_list_next[i] = data_dict_next['atomic_number']
        n_atoms_next = data_dict_next['atomic_number'].shape[0]
        true_atomic_number_list_next[i] = data_dict_next['true_atomic_number']
        position_list_next[i] = data_dict_next['coordinates']
        laf_list_next[i] = torch.cat((data_dict_next['laf'], torch.tensor([1.12]))) #data_dict_next['laf'] #torch.cat((data_dict_next['laf'], torch.zeros(494-n_atoms)))
        edges_u_next[i] =  data_dict_next['edges'][0] + sum_n_atoms_next
        edges_v_next[i] =  data_dict_next['edges'][1] + sum_n_atoms_next
        sum_n_atoms_next += n_atoms_next
        to_jimages_next[i] = data_dict['etype']
        n_atoms_list_next.append(n_atoms_next)

    edges_cat = torch.cat([torch.cat(edges_u)[:,None], torch.cat(edges_v)[:,None]], dim = 1)
    to_jimages = torch.cat(to_jimages, dim = 0)
    ind = torch.where(torch.prod(to_jimages==0, dim = 1))[0]
    edges_cat = edges_cat[ind, :]
    edges_cat = torch.unique(edges_cat, dim = 0)
    
    g = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g.ndata['atomic_number'] = torch.cat(atomic_number_list, dim = 0)
    g.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list, dim = 0)
    g.ndata['position'] = torch.cat(position_list, dim = 0)
    g.lengths_angles_focus = torch.stack(laf_list)#.cuda()
    # g.edata['to_jimages'] = torch.cat(to_jimages, dim = 0)
    g.n_atoms = torch.tensor(n_atoms_list)
    g.inds = torch.tensor(ind_list)

    g_next = dgl.graph(data = torch.unbind(edges_cat, dim = 1), num_nodes = sum_n_atoms)#.to(device='cpu')
    g_next.ndata['atomic_number'] = torch.cat(atomic_number_list_next, dim = 0)
    g_next.ndata['true_atomic_number'] = torch.cat(true_atomic_number_list_next, dim = 0)
    g_next.ndata['position'] = torch.cat(position_list_next, dim = 0)
    g_next.lengths_angles_focus = torch.stack(laf_list_next)#.cuda()
    # g_next.edata['to_jimages'] = torch.cat(to_jimages_next, dim = 0)
    g_next.n_atoms = torch.tensor(n_atoms_list_next)
    g_next.inds = torch.tensor(ind_list)

    actions = torch.tensor([batch[j][1] for j in range(batch_size)])
    rewards = torch.tensor([batch[j][3] for j in range(batch_size)])
    dones = torch.tensor([batch[j][4] for j in range(batch_size)])
    return g, actions, g_next, -rewards, dones

def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    return (frac_coords % 1.)


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


    

    




