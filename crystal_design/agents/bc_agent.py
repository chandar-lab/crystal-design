
from __future__ import annotations
# from .base_agent import BaseAgent
# from crystal_design.agents.qnets.egnn import EGNNetworkBC, GCNetwork
from torch.nn import Linear, Softmax, Sequential
from matgl.layers import MLP, ActivationFunction, BondExpansion, EdgeSet2Set, EmbeddingBlock, MEGNetBlock
from torch import nn
from torch.nn.functional import one_hot
from matgl.models._megnet import MEGNet
from matgl.utils.io import IOMixIn
import dgl
import torch
from torch import nn
from matgl.models._megnet import MEGNet
from matgl.layers import EmbeddingBlock, MLP

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
        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None
        # if activation_type == "swish":
        #     activation = nn.SiLU()
        # elif activation_type == "sigmoid":
        #     activation = nn.Sigmoid()
        # elif activation_type == "tanh":
        #     activation = nn.Tanh()
        # elif activation_type == "softplus2":
        #     activation = SoftPlus2()
        # elif activation_type == "softexp":
        #     activation = SoftExponential()
        # else:
        #     raise Exception("Undefined activation type, please try using swish, sigmoid, tanh, softplus2, softexp")
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
        state_feat = state_feat[None, :]
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
