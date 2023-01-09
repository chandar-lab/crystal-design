from functools import partial
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from hive.agents.qnets.noisy_linear import NoisyLinear
from hive.utils.utils import ActivationFn
from dgl.nn import EGNNConv
from crystal_design.utils.converters.pyg_graph_to_tensor import PyGGraphToTensorConverter

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
        hidden_units: Union[int, List[int]] = [256, 256, 256],
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
        self.egnn_fn_1 = EGNNConv(in_size, hidden_units[0], out_size)
        self.egnn_fn_2 = EGNNConv(out_size, hidden_units[1], out_size)
        self.egnn_fn_3 = EGNNConv(out_size, hidden_units[2], out_size)

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
        h_x = torch.concat([h,x])
        # q_vals = self.linear(h_x)
        return h_x.reshape((1,-1))