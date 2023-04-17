from functools import partial
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from hive.agents.qnets.noisy_linear import NoisyLinear
from hive.utils.utils import ActivationFn
from dgl.nn import EGNNConv
from dgl.nn.pytorch.conv import GraphConv
from crystal_design.utils.data_utils import build_crystal, frac_to_cart_coords
from crystal_design.utils.converters.pyg_graph_to_tensor import PyGGraphToTensorConverter
class GCNetwork(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int, 
        # action_size: int,
        # hidden_units: Union[int, List[int]] = [256, 256],  ## CHANGED THIS
        # activation_fn: ActivationFn = None,
        # noisy: bool = False,
        # std_init: float = 0.5,
        agg: str = 'mean',
    ):  
        super().__init__()
        self.gcn_layer_1 = GraphConv(in_size, out_size * 2).cuda()
        self.gcn_layer_2 = GraphConv(out_size * 2, out_size * 2).cuda()
        self.gcn_layer_3 = GraphConv(out_size * 2, out_size * 2).cuda()
        self.gcn_layer_4 = GraphConv(out_size * 2, out_size).cuda()
        self.agg = agg

    def forward(self, g):
        h = g.ndata['atomic_number']
        x = g.ndata['position'].to(dtype = torch.float32)
        h_x = torch.cat([h,x], dim = 1)  # n x 60
        f1 = self.gcn_layer_1(g, h_x) 
        f2 = self.gcn_layer_2(g, f1)
        f3 = self.gcn_layer_3(g, f2)
        h_x = self.gcn_layer_4(g, f3)  # n x 128

        if self.agg == 'mean':
            h_x = h_x.reshape((-1, 5, h_x.shape[-1]))
            h_x = torch.mean(h_x, dim = 1)

        return h_x

class EGNNetwork(nn.Module):
    """Basic MLP neural network architecture.
    Contains a series of :py:class:`torch.nn.Linear` or
    :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear` layers, each of which
    is followed by a ReLU.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int, 
        # action_size: int,
        hidden_units: Union[int, List[int]] = [256, 256],  ## CHANGED THIS
        # activation_fn: ActivationFn = None,
        # noisy: bool = False,
        # std_init: float = 0.5,
        agg: str = 'mean',
        config: dict = {'max_num_nodes': 5, 'max_num_edges':25, 'node_ftr_dim': 57, 'to_numpy': False}
    ):
        """
        Args:
            in_dim (tuple[int]): The shape of input observations.
            hidden_units (int | list[int]): The number of neurons for each mlp layer.
            noisy (bool): Whether the MLP should use
                :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear` layers or normal
                :py:class:`torch.nn.Linear` layers.
            std_init (float): The range for the initialization of the standard deviation of the
                weights in :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear`.
        """
        super().__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        # if activation_fn is None:
        #     activation_fn = torch.nn.ReLU
        self.egnn_fn_1 = EGNNConv(in_size, hidden_units[0], out_size).cuda()
        self.egnn_fn_2 = EGNNConv(out_size, hidden_units[1], out_size).cuda()
        self.egnn_fn_3 = EGNNConv(out_size, hidden_units[2], out_size).cuda()
        self.egnn_fn_4 = EGNNConv(out_size, hidden_units[3], out_size).cuda()

        # modules = [egnn_fn]
        # for i in range(len(hidden_units) - 1):
            # modules.append(EGNNConv(out_size, hidden_units[i + 1], out_size))
        # self.network = torch.nn.Sequential(*modules)
        # self.modules = torch.nn.Sequential(*modules)
        self.agg = agg
        self.converter = PyGGraphToTensorConverter(config)
        # self.linear = torch.nn.Linear(out_size + 3, action_size)
    def forward(self, rep):
        # x = x.float()
        # x = torch.flatten(x, start_dim=1)
        g = self.converter.decode(rep)
        h = g.ndata['atomic_number']
        x = g.ndata['position']
        # for egnn_layer in self.modules:
        #     h, x = egnn_layer(g, h, x)
        h, x = self.egnn_fn_1(g, h, x)
        h, x = self.egnn_fn_2(g, h, x)
        h, x = self.egnn_fn_3(g, h, x)

        if self.agg == 'mean':
            h = torch.mean(h, dim = 0)
            x = torch.mean(x, dim = 0)
        elif self.agg == 'sum':
            h = torch.sum(h, dim = 0)
            x = torch.sum(x, dim = 0)
        elif self.agg == 'identity':
            pass
        h_x = torch.concat([h,x])
        # q_vals = self.linear(h_x)
        return h_x.reshape((1,-1))
        
# class EGNNetwork(nn.Module):
#     """Basic MLP neural network architecture.
#     Contains a series of :py:class:`torch.nn.Linear` or
#     :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear` layers, each of which
#     is followed by a ReLU.
#     """

#     def __init__(
#         self,
#         in_size: int,
#         out_size: int, 
#         # action_size: int,
#         hidden_units: Union[int, List[int]] = [256, 256],  ## CHANGED THIS
#         # activation_fn: ActivationFn = None,
#         # noisy: bool = False,
#         # std_init: float = 0.5,
#         agg: str = 'mean',
#         config: dict = {'max_num_nodes': 5, 'max_num_edges':25, 'node_ftr_dim': 57, 'to_numpy': False}
#     ):
#         """
#         Args:
#             in_dim (tuple[int]): The shape of input observations.
#             hidden_units (int | list[int]): The number of neurons for each mlp layer.
#             noisy (bool): Whether the MLP should use
#                 :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear` layers or normal
#                 :py:class:`torch.nn.Linear` layers.
#             std_init (float): The range for the initialization of the standard deviation of the
#                 weights in :py:class:`~hive.agents.qnets.noisy_linear.NoisyLinear`.
#         """
#         super().__init__()
#         if isinstance(hidden_units, int):
#             hidden_units = [hidden_units]
#         # if activation_fn is None:
#         #     activation_fn = torch.nn.ReLU
#         self.egnn_fn_1 = EGNNConv(in_size, hidden_units[0], out_size).cuda()
#         self.egnn_fn_2 = EGNNConv(out_size, hidden_units[1], out_size).cuda()
#         self.egnn_fn_3 = EGNNConv(out_size, hidden_units[2], out_size).cuda()
#         self.egnn_fn_4 = EGNNConv(out_size, hidden_units[3], out_size).cuda()

#         # modules = [egnn_fn]
#         # for i in range(len(hidden_units) - 1):
#             # modules.append(EGNNConv(out_size, hidden_units[i + 1], out_size))
#         # self.network = torch.nn.Sequential(*modules)
#         # self.modules = torch.nn.Sequential(*modules)
#         self.agg = agg
#         self.converter = PyGGraphToTensorConverter(config)
#         # self.linear = torch.nn.Linear(out_size + 3, action_size)
#     def forward(self, rep):
#         # x = x.float()
#         # x = torch.flatten(x, start_dim=1)
#         g = self.converter.decode(rep)
#         h = g.ndata['atomic_number']
#         x = g.ndata['position']
#         # for egnn_layer in self.modules:
#         #     h, x = egnn_layer(g, h, x)
#         h, x = self.egnn_fn_1(g, h, x)
#         h, x = self.egnn_fn_2(g, h, x)
#         h, x = self.egnn_fn_3(g, h, x)

#         if self.agg == 'mean':
#             h = torch.mean(h, dim = 0)
#             x = torch.mean(x, dim = 0)
#         elif self.agg == 'sum':
#             h = torch.sum(h, dim = 0)
#             x = torch.sum(x, dim = 0)
#         elif self.agg == 'identity':
#             pass
#         h_x = torch.concat([h,x])
#         # q_vals = self.linear(h_x)
#         return h_x.reshape((1,-1))

class EGNNetworkBC(EGNNetwork):
    def __init__(
        self,
        in_size: int,
        out_size: int, 
        # action_size: int,
        hidden_units: Union[int, List[int]] = [512, 512, 512, 512],
        graph_type: str = 'g',
        # activation_fn: ActivationFn = None,
        # noisy: bool = False,
        # std_init: float = 0.5,
        agg: str = 'mean',
        config: dict = {'max_num_nodes': 5, 'max_num_edges':25, 'node_ftr_dim': 57, 'to_numpy': False}
    ):
        super().__init__(
            in_size = in_size,
            out_size = out_size, 
            # action_size: int,
            hidden_units = hidden_units,
            # activation_fn: ActivationFn = None,
            # noisy: bool = False,
            # std_init: float = 0.5,
            agg = agg,
            config = config,
        )
        self.graph_type = graph_type
        # self.process_h_x = nn.Linear(out_size + 3, 64, device = 'cuda:0')

    def forward(self, g, coord_is_cart = True):
        h = g.ndata['atomic_number']
        if coord_is_cart:
            x = g.ndata['position'].to(dtype = torch.float32)
        else:
            x = frac_to_cart_coords(g.ndata['frac_coords'].to(dtype = torch.float32), g.lengths_angles_focus[:,:3], g.lengths_angles_focus[:,3:6], g.num_nodes)
        # h1, x1 = self.egnn_fn_1(g, h, x)
        # h2, x2 = self.egnn_fn_2(g, h1, x1)
        # h3, x3 = self.egnn_fn_3(g, h2, x2)
        # h4, x4 = self.egnn_fn_4(g, h3, x3)

        # h_final = torch.stack([h1,h2,h3,h4])   ### 4 x N x 128
        # x_final = torch.stack([x1,x2,x3,x4])  ### 4 x N x 3

        # h_x_final = self.process_h_x(torch.cat([h_final, x_final], dim = 2)) ### 4 x N x 64

        # h_x = torch.sum(h_x_final, dim = 0) # N x 64
        # if self.graph_type == 'mg':
            # e = g.edata['to_jimages']
            # h, x = self.egnn_fn_4(g, *self.egnn_fn_3(g, *self.egnn_fn_2(g, *self.egnn_fn_1(g, h,x, e), e), e), e)
        # else:
        h, x = self.egnn_fn_4(g, *self.egnn_fn_3(g, *self.egnn_fn_2(g, *self.egnn_fn_1(g, h, x))))
        if self.agg == 'mean':
            h = h.reshape((-1, 5, h.shape[-1]))
            x = x.reshape((-1, 5, x.shape[-1]))
            h = torch.mean(h, dim = 1)
            x = torch.mean(x, dim = 1)

        if self.agg == 'flatten':
            h_x = torch.concat([h,x], dim = -1)
            h_x = h_x.reshape((h_x.shape[0] // 5, 5 * h_x.shape[1]))
            return h_x
            
        elif self.agg == 'sum':
            h = h.reshape((-1, 5, h.shape[-1]))
            x = x.reshape((-1, 5, x.shape[-1]))
            h = torch.sum(h, dim = 1)
            x = torch.sum(x, dim = 1)
        
        h_x = torch.concat([h,x], dim = -1)

        # q_vals = self.linear(h_x)
        return h_x#.reshape((1,-1))

# class EGNNetworkBC(EGNNetwork):
#     def __init__(
#         self,
#         in_size: int,
#         out_size: int, 
#         # action_size: int,
#         hidden_units: Union[int, List[int]] = [256, 256, 256, 256],
#         graph_type: str = 'g',
#         # activation_fn: ActivationFn = None,
#         # noisy: bool = False,
#         # std_init: float = 0.5,
#         agg: str = 'mean',
#         config: dict = {'max_num_nodes': 5, 'max_num_edges':25, 'node_ftr_dim': 57, 'to_numpy': False}
#     ):
#         super().__init__(
#             in_size = in_size,
#             out_size = out_size, 
#             # action_size: int,
#             hidden_units = hidden_units,
#             # activation_fn: ActivationFn = None,
#             # noisy: bool = False,
#             # std_init: float = 0.5,
#             agg = agg,
#             config = config,
#         )
#         self.graph_type = graph_type
#         self.output_layer = nn.Linear(259 + 500, 118, device = 'cuda:0')
#         # self.process_h_x = nn.Linear(out_size + 3, 64, device = 'cuda:0')

#     def forward(self, g, coord_is_cart = True):
#         h = g.ndata['atomic_number']
#         if coord_is_cart:
#             x = g.ndata['position'].to(dtype = torch.float32)
#         else:
#             x = frac_to_cart_coords(g.ndata['frac_coords'].to(dtype = torch.float32), g.lengths_angles_focus[:,:3], g.lengths_angles_focus[:,3:6], g.num_nodes)
#         # h1, x1 = self.egnn_fn_1(g, h, x)
#         # h2, x2 = self.egnn_fn_2(g, h1, x1)
#         # h3, x3 = self.egnn_fn_3(g, h2, x2)
#         # h4, x4 = self.egnn_fn_4(g, h3, x3)

#         # h_final = torch.stack([h1,h2,h3,h4])   ### 4 x N x 128
#         # x_final = torch.stack([x1,x2,x3,x4])  ### 4 x N x 3

#         # h_x_final = self.process_h_x(torch.cat([h_final, x_final], dim = 2)) ### 4 x N x 64

#         # h_x = torch.sum(h_x_final, dim = 0) # N x 64
#         # if self.graph_type == 'mg':
#             # e = g.edata['to_jimages']
#             # h, x = self.egnn_fn_4(g, *self.egnn_fn_3(g, *self.egnn_fn_2(g, *self.egnn_fn_1(g, h,x, e), e), e), e)
#         # else:
#         h, x = self.egnn_fn_1(g, h, x)
#         # h, x = self.egnn_fn_4(g, *self.egnn_fn_3(g, *self.egnn_fn_2(g, *self.egnn_fn_1(g, h, x))))
#         if self.agg == 'mean':
#             h_split = torch.split(h, g.n_atoms.numpy().tolist())
#             x_split = torch.split(x, g.n_atoms.numpy().tolist())
#             h = [torch.mean(split, dim=0) for split in h_split]
#             x = [torch.mean(split, dim=0) for split in x_split]
#             h = torch.stack(h)
#             x = torch.stack(x)

#         if self.agg == 'flatten':
#             h_x = torch.concat([h,x], dim = -1)
#             h_x = h_x.reshape((h_x.shape[0] // 5, 5 * h_x.shape[1]))
#             return h_x
            
#         elif self.agg == 'sum':
#             h = h.reshape((-1, 5, h.shape[-1]))
#             x = x.reshape((-1, 5, x.shape[-1]))
#             h = torch.sum(h, dim = 1)
#             x = torch.sum(x, dim = 1)
        
#         h_x = torch.concat([h,x], dim = -1)
#         q_values = self.output_layer(torch.cat([h_x, g.lengths_angles_focus], dim = 1))
#         # q_vals = self.linear(h_x)
#         if torch.any(torch.isnan(q_values)):
#             breakpoint()
#         return q_values#.reshape((1,-1))


class EGNNetworkOffline(EGNNetwork):
    def __init__(
        self,
        in_size: int,
        out_size: int, 
        # action_size: int,
        hidden_units: Union[int, List[int]] = [512, 512, 512, 512],
        graph_type: str = 'g',
        # activation_fn: ActivationFn = None,
        # noisy: bool = False,
        # std_init: float = 0.5,
        agg: str = 'mean',
        config: dict = {'max_num_nodes': 5, 'max_num_edges':25, 'node_ftr_dim': 57, 'to_numpy': False}
    ):
        super().__init__(
            in_size = in_size,
            out_size = out_size, 
            # action_size: int,
            hidden_units = hidden_units,
            # activation_fn: ActivationFn = None,
            # noisy: bool = False,
            # std_init: float = 0.5,
            agg = agg,
            config = config,
        )
        self.graph_type = graph_type
        # self.process_h_x = nn.Linear(out_size + 3, 64, device = 'cuda:0')

    def forward(self, g, coord_is_cart = True):
        h = g.ndata['atomic_number']
        if coord_is_cart:
            x = g.ndata['position'].to(dtype = torch.float32)
        else:
            x = frac_to_cart_coords(g.ndata['frac_coords'].to(dtype = torch.float32), g.lengths_angles_focus[:,:3], g.lengths_angles_focus[:,3:6], g.num_nodes)
        # h1, x1 = self.egnn_fn_1(g, h, x)
        # h2, x2 = self.egnn_fn_2(g, h1, x1)
        # h3, x3 = self.egnn_fn_3(g, h2, x2)
        # h4, x4 = self.egnn_fn_4(g, h3, x3)

        # h_final = torch.stack([h1,h2,h3,h4])   ### 4 x N x 128
        # x_final = torch.stack([x1,x2,x3,x4])  ### 4 x N x 3

        # h_x_final = self.process_h_x(torch.cat([h_final, x_final], dim = 2)) ### 4 x N x 64

        # h_x = torch.sum(h_x_final, dim = 0) # N x 64
        # if self.graph_type == 'mg':
            # e = g.edata['to_jimages']
            # h, x = self.egnn_fn_4(g, *self.egnn_fn_3(g, *self.egnn_fn_2(g, *self.egnn_fn_1(g, h,x, e), e), e), e)
        # else:
        h, x = self.egnn_fn_4(g, *self.egnn_fn_3(g, *self.egnn_fn_2(g, *self.egnn_fn_1(g, h, x))))
        if self.agg == 'mean':
            h = h.reshape((-1, 5, h.shape[-1]))  ### CHANGE THIS
            x = x.reshape((-1, 5, x.shape[-1]))
            h = torch.mean(h, dim = 1)
            x = torch.mean(x, dim = 1)

        if self.agg == 'flatten':
            h_x = torch.concat([h,x], dim = -1)
            h_x = h_x.reshape((h_x.shape[0] // 5, 5 * h_x.shape[1]))
            return h_x
            
        elif self.agg == 'sum':
            h = h.reshape((-1, 5, h.shape[-1]))
            x = x.reshape((-1, 5, x.shape[-1]))
            h = torch.sum(h, dim = 1)
            x = torch.sum(x, dim = 1)
        
        h_x = torch.concat([h,x], dim = -1)

        # q_vals = self.linear(h_x)
        return h_x#.reshape((1,-1))