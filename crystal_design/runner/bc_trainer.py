import argparse
import os
from crystal_design.agents.bc_agent import EGNNAgentBC, RandomAgent, GCNAgentBC, LinearAgentBC
import numpy as np
import torch
from torch import nn
import dgl
from tqdm import tqdm 
import wandb
from crystal_design.utils import collate_function, collate_functionV2, collate_functionV3
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
import argparse

class BC_Trainer(object):
    def __init__(self, agent, load_initial_expertdata = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/traj_dict.pt', 
                load_val_expertdata = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/val_traj_dict.pt',
                batch_size = 256, val_batch_size = 16, epochs = 1000, learning_rate = 1e-3, save_path = None, supervise = False, graph_type = 'mg'):
        # 1) Initialize params
        # 2) Initialize BC agent
        self.agent = agent(graph_type = graph_type) #EGNNAgentBC()
        # self.agent = RandomAgent()
        self.load_initial_expertdata = load_initial_expertdata
        self.load_val_expertdata = load_val_expertdata
        self.batch_size=batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.graph_type = graph_type
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr = self.learning_rate, weight_decay = 1e-5)
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.save_path = save_path

    def run_training_loop(self, n_iter = 1, num_traj_per_crystal = 5, num_val_samples = None):
        for itr in range(n_iter):
            # 1) Collect expert trajectories
            # 2) Train agent 
            if itr == 0:
                train_loader, val_loader = self.collect_trajectories(self.load_initial_expertdata, self.load_val_expertdata, num_traj_per_crystal)
            self.history = self.train_agent(train_loader, val_loader, self.epochs)

    def collect_trajectories(self, load_initial_expertdata, load_val_expertdata, num_traj_per_crystal = 5, num_val_samples = None):
        print("\nCollecting data to be used for training...")
        train_data = torch.load(load_initial_expertdata)#['data']
        val_data = torch.load(load_val_expertdata)#['data']
        if self.graph_type == 'g':
            collate_fn = collate_functionV2
        elif self.graph_type == 'mg':
            collate_fn = collate_functionV3
        if num_traj_per_crystal < 5:
            small_train_data = []
            n = len(train_data)
            for i in range(0,n,25):
                indices = np.random.choice(range(5), size = num_traj_per_crystal, replace = False)
                all_traj_crys = train_data[i:i+25]
                for j in indices:
                    small_train_data += all_traj_crys[5*j:5*j+5]
            n = len(val_data)
            small_val_data = []
            for i in range(0,n,25):
                indices = np.random.choice(range(5), size = num_traj_per_crystal, replace = False)
                all_traj_crys_val = val_data[i:i+25]
                for j in indices:
                    small_val_data += all_traj_crys_val[5*j:5*j+5]
            train_loader = DataLoader(small_train_data, batch_size = self.batch_size, collate_fn = collate_fn, shuffle = True, num_workers = 16)
            val_loader = DataLoader(small_val_data, batch_size = self.val_batch_size, collate_fn = collate_fn, shuffle = True, num_workers = 16)
        else:
            train_loader = DataLoader(train_data, batch_size = self.batch_size, collate_fn = collate_fn, shuffle = True, num_workers = 16)
            val_loader = DataLoader(val_data, batch_size = self.val_batch_size, collate_fn = collate_fn, shuffle = True, num_workers = 16)

        return train_loader, val_loader #loaded_paths, actions, lengths_angles_focus, val_paths, val_actions, val_lengths_angles_focus

    def train_agent(self, train_loader, val_loader, epochs):
        history = {'train_loss': [], 'val_loss':[]}
        print('\nTraining agent using sampled data from replay buffer...')
        VAL_EVAL_LIST = []
        for i in tqdm(range(1, epochs+1)):
            step_loss_list = []
            step_valloss_list = []
            val_acc_list = []
            train_acc_list = []
            for train_batch in (train_loader):
                train_batch = train_batch.to(device = 'cuda:0')
                train_batch.action = train_batch.action.to(device = 'cuda:0')
                train_batch.lengths_angles_focus = train_batch.lengths_angles_focus.to(device = 'cuda:0')
                self.optimizer.zero_grad()
                action_distribution = self.agent(train_batch)
                loss = self.loss(action_distribution, train_batch.action)
                train_acc = torch.mean((torch.max(action_distribution, dim = 1).indices == train_batch.action).float())
                train_acc_list.append(train_acc)
                step_loss_list.append(loss)
                loss.backward()
                self.optimizer.step()
            avg_train_loss = torch.mean(torch.stack(step_loss_list))
            avg_train_acc = torch.mean(torch.stack(train_acc_list))
            # if i % 10 == 0:
                # torch.save(self.agent, self.save_path + '/model_' + str(i) + '.pt')
            
            for val_batch in (val_loader):
                with torch.no_grad():
                    val_batch = val_batch.to(device = 'cuda:0')
                    val_batch.action = val_batch.action.to(device = 'cuda:0')
                    val_batch.lengths_angles_focus = val_batch.lengths_angles_focus.to(device = 'cuda:0')
                    val_action_dist = self.agent(val_batch)
                    val_loss = self.loss(val_action_dist, val_batch.action)
                    val_acc = torch.mean((torch.max(val_action_dist, dim = 1).indices == val_batch.action).float())
                    # VAL_EVAL_LIST.append((val_action_dist, val_batch.action, val_loss, val_acc))
                step_valloss_list.append(val_loss)
                val_acc_list.append(val_acc)
            avg_val_loss = torch.mean(torch.stack(step_valloss_list))
            avg_val_acc = torch.mean(torch.stack(val_acc_list))
            wandb.log({'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'epoch': i, 'train_acc': avg_train_acc, 'val_acc': avg_val_acc}, step = i)
            # print('Epoch ' + str(i) + ' Train Loss: ', avg_train_loss, ' ; Val Loss: ', val_loss)
        # torch.save(VAL_EVAL_LIST, 'val_eval_list.pt')
        return history
    
def generate_loop(agent, data_path, save_path):
    data = torch.load(data_path)
    n = len(data)
    all_focus_list = []
    data_new = []

    for i in range(n):
        all_focus_list.append(torch.argmax(data[i]['laf'][6:]))
        if i % 5 == 0:
            data_new.append(data[i])
    all_focus = torch.stack(all_focus_list).reshape((-1,5))  ### n / 5 x 5
    loader = DataLoader(data_new, batch_size = 128, collate_fn = collate_functionV2, num_workers = 0)
    loader_focus = DataLoader(all_focus, batch_size = 128, num_workers = 0)
    data_list = []
    acc_list = []
    for batch, batch_focus in tqdm(zip(loader, loader_focus)):  ## batch.ndata['atomic_number'] -- N x 57
        index = 0 
        batch_size = batch.action.shape[0]
        row_num = torch.arange(batch_size).unsqueeze(1)
        batch_focus_ind = batch_focus + 5 * row_num
        # batch_focus = deepcopy(all_focus)[]
        batch = batch.to(device = 'cuda:0')
        batch.lengths_angles_focus = batch.lengths_angles_focus.to(device = 'cuda:0')
        
        for index in range(5):
            focus = batch_focus_ind[:, index] ## (batch_size,) #batch.lengths_angles_focus[:,6:]
            action_distn = agent(batch)
            action_indices = torch.max(action_distn, dim = 1).indices
            batch.ndata['atomic_number'][focus, action_indices] = 1.
            batch.ndata['atomic_number'][focus, -1] = 0.
            if index < 4:
                batch.lengths_angles_focus[range(batch_size), 6+batch_focus[:, index]] = 0.
                batch.lengths_angles_focus[range(batch_size), 6+batch_focus[:, index + 1]] = 1.
        data_list.append(batch)
        acc_list.append(torch.mean((torch.argmax(batch.ndata['atomic_number'], dim = 1) == torch.argmax(batch.ndata['true_atomic_number'], dim = 1)).float()))
    test_acc = torch.mean(torch.stack(acc_list))
    print('Test Accuracy: ', test_acc)
    torch.save(data_list, save_path)
    return test_acc

    def sample_trajectories(self,):
        pass

    def sample_trajectory(self,):
        pass

    def logging(self,):
        pass

class BC_Supervise(nn.Module):
    def __init__():
        super().__init__()
    def forward():
        pass

if __name__ == '__main__':
    exp_num = np.random.randint(10000)
    parser = argparse.ArgumentParser(description='Behavioral Cloning - Crystal Design')
    parser.add_argument('--exp_num', type = int, default = 0, help = 'exp_num')
    parser.add_argument('--num_traj_per_crystal', type = int, default = 5, help = 'num_traj_per_crystal')
    parser.add_argument('--agent', type = str, default = 'EGNNAgentBC', help = 'agent')
    parser.add_argument('--gnn_hidden', type = list, default = [256, 256, 256, 256])
    parser.add_argument('--batch_size', type = int, default = 2048)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--epochs', type = int, default = 1000)
    parser.add_argument('--graph_type', type = str, default='g')
    parser.add_argument('--agg', type = str, default = 'mean')
    parser.add_argument('--wandb', type = bool, default = True)
    args = parser.parse_args()
    agent_name = args.agent
    if agent_name == 'GCNAgentBC':
        agent = GCNAgentBC
    elif agent_name == 'EGNNAgentBC':
        agent = EGNNAgentBC
    elif agent_name == 'RandomAgent':
        agent = RandomAgent
    elif agent_name == 'LinearAgentBC':
        agent = LinearAgentBC

    wandb.login()
    exp_num = args.exp_num 
    print('Starting Run ', exp_num)
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_traj_per_crystal = args.num_traj_per_crystal
    graph_type = args.graph_type
    agg = args.agg
    epochs = args.epochs
    if graph_type == 'mg':
        path = agg + '_' + agent_name + '_MGBC_11kx' + str(num_traj_per_crystal) + '_' + str(batch_size) + 'b' + str(learning_rate) + 'lr' + '1e-5wd' + 'adam_4layer' + '_' + str(epochs) + 'conf'
    else:
        path = agg + '_' + agent_name + '_GBC_11kx' + str(num_traj_per_crystal) + '_' + str(batch_size) + 'b' + str(learning_rate) + 'lr' + '1e-5wd' + 'adam_4layer' + '_' + str(epochs)  + 'conf'

    project = 'CRYSTAL-BC'
    group = path + '_256h'
    name = "Run" + str(exp_num)
    wandb_log = args.wandb
    if wandb_log:
        wandb.init(config  = vars(args),
                    project = project,
                    group = group,
                    name = name)
    # path = 'bc_11kx5kv3_random_agent' # + str(batch_size) + 'b' + str(learning_rate) + 'lr' + '1e-5wd' + str(exp_num) + 'adam_4layer'
    path = path + str(exp_num)
    if graph_type == 'mg':
        try:
            os.mkdir('../models/perov_mg_bc/' + path)
        except:
            os.rmdir('../models/perov_mg_bc/' + path)
            os.mkdir('../models/perov_mg_bc/' + path)
    else:
        try:
            os.mkdir('../models/perov_bc/' + path)
        except:
            os.rmdir('../models/perov_bc/' + path)
            os.mkdir('../models/perov_bc/' + path)
    seeds = [1234, 3422, 2354, 6454, 3209]
    torch.manual_seed(seeds[exp_num])
    bc_trainer = BC_Trainer(agent = agent, learning_rate = learning_rate, epochs=epochs, batch_size=batch_size, val_batch_size=512,
                load_initial_expertdata = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/train_tens_dict.pt', #'/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/train_traj_dict_new_new.pt',
                load_val_expertdata='/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/val_tens_dict.pt', #'/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/val_traj_dict_new_new.pt',
                save_path = path, graph_type = 'g')
    bc_trainer.run_training_loop(num_traj_per_crystal = num_traj_per_crystal)

    # for traj in [5]:
    #     test_acc_list = []
    #     for run in range(5):
    #         # path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/models/perov_bc/final_bc_11kx'+str(traj)+'2048b0.001lr1e-5wdadam_4layer_run' + str(run) + '/'
    #         # 5_2048b0.001lr1e-5wdadam_4layer_run0
    #         # path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/models/perov_bc/GCNAgentBC_bc_11kx'+str(traj)+'_2048b0.001lr1e-5wdadam_4layer_run' + str(run) + '/'
    #         # path  = 'random_agent_' + str(run)
    #         path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/models/perov_bc/final_bc_11kx52048b0.001lr1e-5wdadam_4layer_run' + str(run) + '/'
    #         agent_model = torch.load(path + 'model_1000.pt')
    #         # agent_model = RandomAgent()
    #         data_path = '/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/trajectories/val_tens_dict.pt'
    #         try:
    #             os.mkdir(path)
    #         except:
    #             pass
    #         save_path = path +'bc_gen_val.pt'
    #         test_acc = generate_loop(agent_model, data_path, save_path)
    #         test_acc_list.append(test_acc)
    #     print(str(traj) + ' Traj Val Accuracy: ', torch.mean(torch.stack(test_acc_list)))
    