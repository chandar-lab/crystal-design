# from crystal_design.replays.replay_buffer import ReplayBuffer
# from hw1.roble.infrastructure.replay_buffer import ReplayBuffer
# from hw1.roble.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent
from crystal_design.agents.qnets.egnn import EGNNetworkBC, GCNetwork
from torch.nn import Linear, Softmax, Sequential
from torch import nn
from torch.nn.functional import one_hot
import torch


class BCAgent(BaseAgent):
    def __init__(self, agent_params):
        super(BCAgent, self).__init__()
        # 1) Initialize parameters
        # 2) Initilalize policy network
        # 3) Initialize replay buffer

        self.agent_params = agent_params ## Initialize parameters

        # actor/policy   ## Initialize Policy Network
        self.actor = EGNNetwork(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['network_width'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):   
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        log = self.actor.update(ob_no, ac_na)  # HW1: you will modify this
        return log

    def add_to_replay_buffer(self, paths):  # Not important as of now
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size): 
        return self.replay_buffer.sample_random_data(batch_size)  # HW1: you will modify this

    def save(self, path):
        return self.actor.save(path)


class EGNNAgentBC(nn.Module):
    def __init__(self,in_dim_node = 57, out_dim_node = 256, in_dim_graph = 11, out_dim_graph = 8, graph_type = 'g', agg = 'mean'):
        super().__init__()
        self.egnn_net = EGNNetworkBC(in_dim_node, out_dim_node, graph_type=graph_type)
        self.prop_layer = Linear(in_dim_graph, out_dim_graph, device = 'cuda:0') ## Add activation
        if agg == 'flatten':
            self.output_layer = Linear((out_dim_node + 3) * 5 + out_dim_graph, in_dim_node - 1, device = 'cuda:0')
        elif agg == 'mean':
            self.output_layer = Linear((out_dim_node + 3) + out_dim_graph, in_dim_node - 1, device = 'cuda:0')
        self.graph_type = graph_type
        # self.output_layer = Linear(64 + out_dim_graph, in_dim_node - 1, device = 'cuda:0')

    def forward(self, g):
        h_x = self.egnn_net(g)  ## b x (5 * 131)
        l_a_f = g.lengths_angles_focus
        p = self.prop_layer(l_a_f)
        p = nn.ReLU()(p)
        output = torch.cat([h_x, p], dim = 1)            
        output = self.output_layer(output)
        return output


# class EGNNAgentBC(nn.Module):
#     def __init__(self,in_dim_node = 118, out_dim_node = 256, in_dim_graph = 11, out_dim_graph = 8, graph_type = 'g', agg = 'mean'):
#         super().__init__()
#         self.egnn_net = EGNNetworkBC(in_dim_node, out_dim_node, graph_type=graph_type)
#         self.prop_layer = Linear(in_dim_graph, out_dim_graph, device = 'cuda:0') ## Add activation
#         if agg == 'flatten':
#             self.output_layer = Linear((out_dim_node + 3) * 5 + out_dim_graph, in_dim_node - 1, device = 'cuda:0')
#         elif agg == 'mean':
#             self.output_layer = Linear((out_dim_node + 3) + out_dim_graph, in_dim_node - 1, device = 'cuda:0')
#         self.graph_type = graph_type
#         # self.output_layer = Linear(64 + out_dim_graph, in_dim_node - 1, device = 'cuda:0')

#     def forward(self, g):
#         h_x = self.egnn_net(g)  ## b x (5 * 131)
#         # l_a_f = g.lengths_angles_focus
#         # p = self.prop_layer(l_a_f)
#         output = (h_x)
#         # output = torch.cat([h_x, p], dim = 1)            
#         # output = self.output_layer(output)
#         return output

class GCNAgentBC(nn.Module):
    def __init__(self,in_dim_node = 57, out_dim_node = 128, in_dim_graph = 11, out_dim_graph = 8, graph_type = 'g'):
        super().__init__()
        self.egnn_net = GCNetwork(in_dim_node + 3, out_dim_node)
        self.prop_layer = Linear(in_dim_graph, out_dim_graph, device = 'cuda:0') ## Add activation
        self.output_layer = Linear(out_dim_node + out_dim_graph, in_dim_node - 1, device = 'cuda:0')
        # self.output_layer = Linear(64 + out_dim_graph, in_dim_node - 1, device = 'cuda:0')

    def forward(self, g):
        h_x = self.egnn_net(g)  ## n x 64
        l_a_f = g.lengths_angles_focus
        p = self.prop_layer(l_a_f)
        p = nn.ReLU()(p)
        output = torch.cat([h_x, p], dim = 1)            
        output = self.output_layer(output)
        return output

class LinearAgentBC(nn.Module):
    def __init__(self, in_dim_node = (57 + 3) * 5, in_dim_graph = 11, out_dim_graph = 8, out_dim_node = 128, hidden = [256,256,256], n_vocab = 56):
        super().__init__()
        self.linear_layers = Sequential(Linear(in_dim_node, hidden[0], device = 'cuda:0'), nn.ReLU(), Linear(hidden[0], hidden[1], device = 'cuda:0'), nn.ReLU(),
                                        Linear(hidden[1], hidden[2], device = 'cuda:0'), nn.ReLU(), Linear(hidden[2], out_dim_node, device = 'cuda:0'))
        self.prop_layer = Linear(in_dim_graph, out_dim_graph, device = 'cuda:0') ## Add activation
        self.output_layer = Linear(out_dim_node + out_dim_graph, 56, device = 'cuda:0')

    def forward(self, g):
        l_a_f = g.lengths_angles_focus
        h = g.ndata['atomic_number']
        x = g.ndata['position'].to(dtype = torch.float32)
        h_x = torch.cat([h,x], dim = 1)
        h_x = h_x.reshape((h_x.shape[0] // 5, 5 * h_x.shape[1]))
        h_x = self.linear_layers(h_x)
        p = self.prop_layer(l_a_f)
        p = nn.ReLU()(p)
        output = torch.cat([h_x, p], dim = 1)            
        output = self.output_layer(output)
        return output

class RandomAgent(nn.Module):
    def __init__(self, n_vocab = 56):
        super().__init__()
        self.n_vocab = n_vocab
    def forward(self, batch):
        batch_size = batch.lengths_angles_focus.shape[0]
        return one_hot(torch.randint(low = 0, high = self.n_vocab, size = (batch_size, )), num_classes = self.n_vocab).to(dtype = torch.float32, device = 'cuda:0')