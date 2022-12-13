from crystal_design.utils.converters import BaseConverter
from torch_geometric.data import Data
from torchtyping import TensorType
from typing import Dict
import torch

class PyGGraphToTensorConverter(BaseConverter):
    def __init__(self, config: Dict[str, object]):
        self.max_num_nodes = config['max_num_nodes']
        self.max_num_edges = config['max_num_nodes']

        self.node_ftr_dim = config['node_ftr_dim']
        self.edge_ftr_dim = config['edge_ftr_dim']

        self.padding_token = torch.nan

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

    def _pad_edge_features(self, edge_ftrs: TensorType) -> TensorType
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

    def encode(self, graph: Data) -> TensorType:
        node_ftrs = self._pad_node_features(graph.x)
        edge_idx = self._pad_edge_index(graph.edge_index)
        edge_ftrs = self._pad_edge_features(graph.edge_attr)

        combined_tnsr = torch.cat([
            node_ftrs.flatten(),
            edge_idx.flatten(),
            edge_ftrs.flatten()
        ])

        return (
            combined_tnsr.cpu().numpy() if self.encode_to_numpy else combined_tnsr
        )

    def _decode_tensor(
        self,
        tensor: TensorType,
        slice_idxs: Tuple[int, int],
        shape: Tuple[int, int]
    ) -> TensorType:
        padded_decoded = tensor[slice_idxs[0] : slice_idxs[1]].view(shape)

        not_padding_nnz = (padded_decoded != self.padding_token).nonzero()
        return padded_decoded[not_padding_nnz[:, 0], not_padding_nnz[:, 1]]

    def decode(self, tensor: TensorType) -> TensorType:
        if self.encode_to_numpy:
            tensor = torch.tensor(tensor)

        node_ftr_end_idx = self.max_num_edges * self.node_ftr_dim
        node_ftrs = self._decode_tensor(
            tensor,
            (0, node_ftr_end_idx),
            (self.max_num_nodes, self.node_ftr_dim)
        )

        edge_idx_end_idx = node_ftr_end_idx + (2 * self.max_num_edges)
        edge_idx = self._decode_tensor(
            tensor,
            (node_ftr_end_idx, edge_idx_end_idx),
            (2, self.max_num_edges)
        )

        edge_ftrs_end_idx = tensor.shape[-1]
        edge_ftrs = self._decode_tensor(
            tensor,
            (edge_idx_end_idx, edge_ftrs_end_idx),
            (self.max_num_edges, self.edge_ftr_dim)
        )

        return Data(
            x=node_ftrs if len(node_ftrs) else None,
            edge_index=edge_idx if len(edge_idx) else None,
            edge_attr=edge_ftrs if len(edge_ftrs) else None,
        )
