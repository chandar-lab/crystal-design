# from crystal_design.utils.converters import BaseConverter
from crystal_design.utils.converters.base_converter import BaseConverter
# from torch_geometric.data import Data
import dgl
from dgl.heterograph import DGLHeteroGraph
from torchtyping import TensorType
from typing import Dict
import torch
from typing import Tuple

class PyGGraphToTensorConverter(BaseConverter):
    def __init__(self, config: Dict[str, object]):
        self.max_num_nodes = config['max_num_nodes']
        self.max_num_edges = config['max_num_edges']

        self.node_ftr_dim = config['node_ftr_dim']
        # self.edge_ftr_dim = config['edge_ftr_dim']

        self.padding_token = torch.nan
        self.encode_to_numpy = config['to_numpy']

    def _pad_node_features(self, node_ftrs: TensorType) -> TensorType:
        dev = node_ftrs.device

        node_padding_size = self.max_num_nodes - node_ftrs.shape[0]
        assert node_padding_size >= 0 and node_ftrs.shape[1] == self.node_ftr_dim

        if node_padding_size > 0:
            to_cat = [
                node_ftrs,
                torch.full(
                    (node_padding_size, self.node_ftr_dim),
                    self.padding_token,
                    device=dev
                )
            ]

            node_ftrs = torch.cat(to_cat, dim=0)

        return node_ftrs
    
    def _pad_node_pos(self, node_ftrs: TensorType) -> TensorType:
        dev = node_ftrs.device

        node_padding_size = self.max_num_nodes - node_ftrs.shape[0]
        assert node_padding_size >= 0 and node_ftrs.shape[1] == 3

        if node_padding_size > 0:
            to_cat = [
                node_ftrs,
                torch.full(
                    (node_padding_size, 3),
                    self.padding_token,
                    device=dev
                )
            ]

            node_ftrs = torch.cat(to_cat, dim=0)

        return node_ftrs

    def _pad_edge_index(self, edge_idx: TensorType) -> TensorType:
        edge_idx_padding_size = self.max_num_edges - edge_idx.shape[1]
        assert edge_idx_padding_size >= 0
        if edge_idx_padding_size > 0:
            to_cat = [
                edge_idx,
                torch.full(
                    (2, edge_idx_padding_size),
                    self.padding_token,
                    device=edge_idx.device
                )
            ]

            edge_idx = torch.cat(to_cat, dim=1)

        return edge_idx

    def _pad_edge_features(self, edge_ftrs: TensorType) -> TensorType:
        edge_ftrs_padding_size = self.max_num_edges - edge_ftrs.shape[0]
        assert edge_ftrs_padding_size >= 0 and \
               edge_ftrs.shape[1] == self.edge_ftr_dim

        if edge_ftrs_padding_size > 0:
            to_cat = [
                edge_ftrs,
                torch.full(
                    (edge_ftrs_padding_size, self.edge_ftr_dim),
                    self.padding_token,
                    device=edge_ftrs.device
                )
            ]

            edge_ftrs = torch.cat(to_cat, dim=0)

        return edge_ftrs

    def encode(self, graph: DGLHeteroGraph) -> TensorType:
        node_ftrs = self._pad_node_features(graph.ndata['atomic_number'])
        node_ftrs_true = self._pad_node_features(graph.ndata['true_atomic_number'])
        coords = self._pad_node_pos(graph.ndata['position'])
        edge_idx = self._pad_edge_index(torch.stack(graph.edges()))
        combined_tnsr = torch.cat([
            node_ftrs.flatten(),
            node_ftrs_true.flatten(),
            coords.flatten(),
            edge_idx.flatten(),
        ])
        return (
            combined_tnsr.cpu().numpy() if self.encode_to_numpy else combined_tnsr
        )

    def _decode_tensor_node(
        self,
        tensor: TensorType,
        slice_idxs: Tuple[int, int],
        shape: Tuple[int, int]
    ) -> TensorType:
        padded_decoded = tensor[slice_idxs[0] : slice_idxs[1]].view(shape)
        not_padding_nnz = ~torch.isnan(padded_decoded)#.nonzero() #(padded_decoded != self.padding_token).nonzero()
        return padded_decoded[not_padding_nnz].reshape((-1, shape[-1]))

    def _decode_tensor_edge(
        self,
        tensor: TensorType,
        slice_idxs: Tuple[int, int],
        shape: Tuple[int, int]
    ) -> TensorType:
        padded_decoded = tensor[slice_idxs[0] : slice_idxs[1]].view(shape)
        not_padding_nnz = ~torch.isnan(padded_decoded)#.nonzero() #(padded_decoded != self.padding_token).nonzero()
        return padded_decoded[not_padding_nnz].reshape((2, -1))

    def _decode_tensor_pos(
        self,
        tensor: TensorType,
        slice_idxs: Tuple[int, int],
        shape: Tuple[int, int]
    ) -> TensorType:
        padded_decoded = tensor[slice_idxs[0] : slice_idxs[1]].view(shape)
        not_padding_nnz = ~torch.isnan(padded_decoded)#.nonzero() #(padded_decoded != self.padding_token).nonzero()
        return padded_decoded[not_padding_nnz].reshape((-1, 3))

    def decode(self, tensor: TensorType) -> TensorType:
        tensor = torch.squeeze(tensor).to(device = torch.device('cuda:0'))
        if self.encode_to_numpy:
            tensor = torch.tensor(tensor)
        node_ftr_end_idx = self.max_num_nodes * self.node_ftr_dim
        node_ftrs = self._decode_tensor_node(
            tensor,
            (0, node_ftr_end_idx),
            (self.max_num_nodes, self.node_ftr_dim)
        )

        node_ftr_end_idx_true = node_ftr_end_idx + self.max_num_nodes * self.node_ftr_dim
        node_ftrs_true = self._decode_tensor_node(
            tensor,
            (node_ftr_end_idx, node_ftr_end_idx_true),
            (self.max_num_nodes, self.node_ftr_dim)
        )

        node_pos_end_idx_true = node_ftr_end_idx_true + self.max_num_nodes * 3
        node_pos = self._decode_tensor_pos(
                    tensor,
                    (node_ftr_end_idx_true, node_pos_end_idx_true),
                    (self.max_num_nodes, 3)
                )

        edge_idx_end_idx = node_pos_end_idx_true + (2 * self.max_num_edges)
        edge_idx = self._decode_tensor_edge(
            tensor,
            (node_pos_end_idx_true, edge_idx_end_idx),
            (2, self.max_num_edges)
        )

        num_atoms = node_ftrs.shape[0]
        g = dgl.DGLGraph().to(torch.device('cuda:0'))
        g.add_nodes(num_atoms)
        g.ndata['atomic_number'] = node_ftrs
        g.ndata['true_atomic_number'] = node_ftrs_true
        g.ndata['position'] = node_pos
        g.add_edges(edge_idx[0,:].to(torch.int64), edge_idx[1,:].to(torch.int64))
        
        return g