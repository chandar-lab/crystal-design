import dgl
import torch
from tqdm import tqdm 

def to_tensor_data(data_path, save_path, num_atoms = 5):
    data = torch.load(data_path)['data']
    n = len(data)
    data_list = []
    for i in tqdm(range(n)):
        g = data[i].to(device = 'cpu')
        edges = g.edges()
        d_ = {'atomic_number':None, 'position':None, 'laf':None, 'action':None, 'edges_u':None, 'edges_v':None}
        d_['atomic_number'] = g.ndata['atomic_number']        
        d_['position'] = g.ndata['position']
        d_['laf'] = g.lengths_angles_focus.cpu()
        d_['action'] = g.action.long().cpu()
        d_['edges_u'] = edges[0]
        d_['edges_v'] = edges[1]
        d_['true_atomic_number'] = g.ndata['true_atomic_number']  
        data_list.append(d_)
    torch.save(data_list, save_path)

def to_tensor_data_mg(data_path, save_path, num_atoms = 5):
    data = torch.load(data_path)['data']
    n = len(data)
    data_list = []
    for i in tqdm(range(n)):
        g = data[i].to(device = 'cpu')
        edges = g.edges()
        d_ = {'atomic_number':None, 'frac_coords':None, 'laf':None, 'action':None, 'edges_u':None, 'edges_v':None, 'to_jimages': None}
        d_['atomic_number'] = g.ndata['atomic_number']
        d_['position'] = g.ndata['coords']        
        d_['frac_coords'] = g.ndata['frac_coords']
        d_['laf'] = g.lengths_angles_focus.cpu()
        d_['action'] = g.action.long().cpu()
        d_['edges_u'] = edges[0]
        d_['edges_v'] = edges[1]
        d_['true_atomic_number'] = g.ndata['true_atomic_number'] 
        d_['to_jimages'] = g.edata['to_jimages']
        data_list.append(d_)
    torch.save(data_list, save_path)

if __name__ == '__main__':
    data_path = 'trajectories/train_mg.pt'
    save_path = 'trajectories/train_mg_dict.pt'
    to_tensor_data_mg(data_path=data_path, save_path=save_path)
    data_path = 'trajectories/val_mg.pt'
    save_path = 'trajectories/val_mg_dict.pt'
    to_tensor_data_mg(data_path=data_path, save_path=save_path)
    data_path = 'trajectories/test_mg.pt'
    save_path = 'trajectories/test_mg_dict.pt'
    to_tensor_data_mg(data_path=data_path, save_path=save_path)