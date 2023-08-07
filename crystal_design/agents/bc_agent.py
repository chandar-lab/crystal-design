# from crystal_design.replays.replay_buffer import ReplayBuffer
# from hw1.roble.infrastructure.replay_buffer import ReplayBuffer
# from hw1.roble.policies.MLP_policy import MLPPolicySL
from __future__ import annotations
from .base_agent import BaseAgent
from crystal_design.agents.qnets.egnn import EGNNetworkBC, GCNetwork
from torch.nn import Linear, Softmax, Sequential
from torch import nn
from torch.nn.functional import one_hot
from matgl.models._megnet import MEGNet
from matgl.utils.io import IOMixIn
import dgl
import torch
from torch import nn
from matgl.layers import EmbeddingBlock, SoftPlus2, SoftExponential, MLP


class EmbeddingBlockRL(EmbeddingBlock):
    """Embedding block for generating node, bond and state features."""

    def __init__(
        self,
        degree_rbf: int,
        activation: nn.Module,
        dim_node_embedding: int,
        dim_edge_embedding: int | None = None,
        dim_state_feats: int | None = None,
        ntypes_node: int | None = None,
        include_state: bool = False,
        ntypes_state: int | None = None,
        dim_state_embedding: int | None = None,
    ):
        """
        Args:
            degree_rbf (int): number of rbf
            activation (nn.Module): activation type
            dim_node_embedding (int): dimensionality of node features
            dim_edge_embedding (int): dimensionality of edge features
            dim_state_feats: dimensionality of state features
            ntypes_node: number of node labels
            include_state: Whether to include state embedding
            ntypes_state: number of state labels
            dim_state_embedding: dimensionality of state embedding.
        """
        super().__init__()
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.dim_node_embedding = dim_node_embedding
        self.dim_edge_embedding = dim_edge_embedding
        self.dim_state_feats = dim_state_feats
        self.ntypes_node = ntypes_node
        self.dim_state_embedding = dim_state_embedding
        self.activation = activation
        if ntypes_state and dim_state_embedding is not None:
            self.layer_state_embedding = nn.Embedding(ntypes_state, dim_state_embedding, device = 'cuda')  # type: ignore
        if ntypes_node is not None:
            self.layer_node_embedding = nn.Embedding(ntypes_node, dim_node_embedding, device = 'cuda')
        if dim_edge_embedding is not None:
            dim_edges = [degree_rbf, dim_edge_embedding]
            self.layer_edge_embedding = MLP(dim_edges, activation=activation, activate_last=True, device = 'cuda')

    def forward(self, node_attr, edge_attr, focus_feat, state_attr):
        """Output embedded features.

        Args:
            node_attr: node attribute
            edge_attr: edge attribute
            state_attr: state attribute

        Returns:
            node_feat: embedded node features
            edge_feat: embedded edge features
            state_feat: embedded state features
        """
        if self.ntypes_node is not None:
            node_feat = self.layer_node_embedding(node_attr)
        else:
            node_embed = MLP([node_attr.shape[-1], self.dim_node_embedding], activation=self.activation)
            node_feat = node_embed(node_attr.to(torch.float32))
        if self.dim_edge_embedding is not None:
            edge_feat = self.layer_edge_embedding(edge_attr.to(torch.float32))
        else:
            edge_feat = edge_attr
        if self.include_state is True:
            if self.ntypes_state and self.dim_state_embedding is not None:
                state_feat = self.layer_state_embedding(state_attr)
            elif self.dim_state_feats is not None:
                state_attr = torch.unsqueeze(state_attr, 0)
                state_embed = MLP([state_attr.shape[-1], self.dim_state_feats], activation=self.activation)
                state_feat = state_embed(state_attr.to(torch.float32))
            else:
                state_feat = state_attr
        else:
            state_feat = None
        return node_feat, edge_feat, state_feat



class MEGNetRL(MEGNet, nn.Module, IOMixIn):
    def __init__(self, 
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 1,
        dim_state_embedding: int = 8,
        ntypes_state: int = 20,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple = (64, 32),
        hidden_layer_sizes_conv: tuple = (64, 64, 32),
        hidden_layer_sizes_output: tuple = (32, 16),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        activation_type: str = "softplus2",
        is_classification: bool = False,
        include_state: bool = True,
        dropout: float = 0.0,
        element_types = [],
        bond_expansion: float = 0.0,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        **kwargs,
    
    ):
        # super(MEGNet, self).__init__()
        super(MEGNetRL, self).__init__(dim_edge_embedding = dim_edge_embedding)
        if activation_type == "swish":
            activation = nn.SiLU()
        elif activation_type == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_type == "tanh":
            activation = nn.Tanh()
        elif activation_type == "softplus2":
            activation = SoftPlus2()
        elif activation_type == "softexp":
            activation = SoftExponential()
        else:
            raise Exception("Undefined activation type, please try using swish, sigmoid, tanh, softplus2, softexp")
        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=89,
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )
   
        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding * 2, *hidden_layer_sizes_input]
        
        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)

        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 88],
            activation=activation,
            activate_last=False,
        )
    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ):
        """Forward pass of MEGnet. Executes all blocks.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: State features.

        Returns:
            Prediction
        """
        edge_feat = self.bond_expansion(edge_feat)
        # edge_feat = edge_feat[:,None]
        node_feat = node_feat.to(dtype = torch.int64)
        focus_feat = torch.split(node_feat[:,-1], graph.batch_num_nodes().cpu().tolist())
        focus_feat = torch.tensor([torch.argmax(f) for f in focus_feat]).to(device = 'cuda')
        node_feat = torch.argmax(node_feat[:,:-1], dim = 1)
        node_feat, edge_feat, focus_feat = self.embedding(node_feat, edge_feat, focus_feat)
        edge_feat = self.edge_encoder(edge_feat.to(dtype = torch.float32))
        node_feat = self.node_encoder(node_feat)
        state_feat = torch.cat((state_feat, focus_feat), dim = 1)
        state_feat = self.state_encoder(state_feat.to(dtype = torch.float32))
        
        for block in self.blocks:
            output = block(graph, edge_feat, node_feat, state_feat)
            edge_feat, node_feat, state_feat = output

        node_vec = self.node_s2s(graph, node_feat)
        edge_vec = self.edge_s2s(graph, edge_feat)

        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        state_feat = torch.squeeze(state_feat)

        vec = torch.hstack([node_vec, edge_vec, state_feat])

        if self.dropout:
            vec = self.dropout(vec)  # pylint: disable=E1102

        output = self.output_proj(vec)
        return output

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


# class EGNNAgentBC(nn.Module):
#     def __init__(self,in_dim_node = 57, out_dim_node = 256, in_dim_graph = 11, out_dim_graph = 8, graph_type = 'g', agg = 'mean'):
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
#         l_a_f = g.lengths_angles_focus
#         p = self.prop_layer(l_a_f)
#         p = nn.ReLU()(p)
#         output = torch.cat([h_x, p], dim = 1)            
#         output = self.output_layer(output)
#         return output


class EGNNAgentBC(nn.Module):
    # def __init__(self,in_dim_node = 118, out_dim_node = 256, in_dim_graph = 500, out_dim_graph = 64, graph_type = 'g', agg = 'mean'):
    def __init__(self,in_dim_node = 90, out_dim_node = 32, in_dim_graph = 11, out_dim_graph = 8, graph_type = 'g', agg = 'mean'):
        super().__init__()
        self.egnn_net = EGNNetworkBC(in_dim_node, out_dim_node, graph_type=graph_type)
        # self.prop_layer = Linear(in_dim_graph, out_dim_graph, device = 'cuda:0') ## Add activation
        # if agg == 'flatten':
            # self.output_layer = Linear((out_dim_node + 3) * 5 + out_dim_graph, in_dim_node - 1, device = 'cuda:0')
        # elif agg == 'mean':
            # self.output_layer = Linear((out_dim_node + 3) + out_dim_graph, in_dim_node - 1, device = 'cuda:0')
        self.graph_type = graph_type
        # self.output_layer = Linear(64 + out_dim_graph, in_dim_node - 1, device = 'cuda:0')

    def forward(self, g):
        h_x = self.egnn_net(g)  ## b x (5 * 131)
        # l_a_f = g.lengths_angles_focus
        # p = self.prop_layer(l_a_f)
        output = (h_x)
        # output = torch.cat([h_x, p], dim = 1)            
        # output = self.output_layer(output)
        return output

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
