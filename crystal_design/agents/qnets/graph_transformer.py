from gflownet.model.graph_transformer import GraphTransformerFragGFN
import torch

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MolFragGraphTransformer(GraphTransformerFragGFN):
    def __init__(self, env_ctx, num_emb=64, num_layers=3, num_heads=2):
        super().__init__(env_ctx, num_emb, num_layers, num_heads)
        self.converter = PyGGraphToTensorConverter({
            'max_num_nodes': 9,
            'max_num_edges': 20,
            'node_ftr_dim': 20,
            'edge_ftr_dim': 20,
        })

        self.env_ctx = env_ctx

    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = self.converter.encode(observation)

        cond = torch.ones(self.env_ctx.num_cond_dim, device=get_device())
        graph_action_cat, _ = super().forward(observation, cond)

        logits_list = graph_action_cat.logits
        if graph_action_cat.masks is not None:
            logits_list = [
                logits * mask
                for logits, mask in zip(logits_list, graph_action_cat.masks)
            ]

        return torch.cat(logits_list), graph_action_cat
