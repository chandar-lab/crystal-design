from hive.envs.base import BaseEnv
from crystal_design.utils import get_device
from gflownet.tasks.seh_frag import SEHTask
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphActionType, GraphBuildingEnv
from typing import Tuple
from torchtyping import TensorType
import torch

OBSERVATION_TYPE = None

NUM_MAX_FRAGS = 9

# Always use a turn index of 0 as we're in a single agent env
TURN_IDX = 0

def _wrap_model(model):
    device = get_device()
    model.to(device=device)

    return model, device

class SEHFragmentMoleculeEnvironment(BaseEnv):
    """
    This code is almost entirely ported from the repository found at
    https://github.com/recursionpharma/gflownet.  Credit for the environment
    goes to the authors there, as well as the authors of the original
    GFlowNet paper.

    Sets up a task where the reward is computed using a proxy for
    the binding energy of a molecule to Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet
    paper, see `gflownet.models.bengio2021flow`.
    """

    def __init__(self):
        self.task_obj = SEHTask(
            dataset=[],
            temperature_distribution='NA',
            temperature_parameters=None,
            wrap_model=_wrap_model
        )

        self._env = GraphBuildingEnv()
        self._env_context = FragMolBuildingEnvContext(max_frags=NUM_MAX_FRAGS)

        self.state = self._env.new()
        self.done = False
        self.step_idx = 0
        self.max_len = 8
        self.max_nodes = 8
        self.sanitize_samples = True

        self.device = get_device()
        self.converter = PyGGraphToTensorConverter({
            'max_num_nodes': 5,
            'max_num_edges':25,
            'node_ftr_dim': 57,
            'to_numpy': False
        })

    def reset(self) -> Tuple[OBSERVATION_TYPE, int]:
        self.state = self._env.new()
        self.done = False

        return self.converter.encode(self._get_embedded_state()), TURN_IDX

    def _get_embedded_state(self) -> TensorType[float]:
        singleton_torch_graphs = [self._env_context.graph_to_Data(self.state)]

        return self._env_context.collate(
            singleton_torch_graphs
        ).to(device=self.device)

    def step(self, action: int) -> Tuple[
        OBSERVATION_TYPE,
        float, # Reward
        bool,  # Done
        int,   # Turn (irrelevant for this env)
        dict   # Custom information dict
    ]:
        if not isinstance(action, tuple):
            raise TypeError(
                'Action must be of type tuple, was of type %s' % type(action)
            )

        if self.done:
            raise ValueError(
                'Called step() on an environment which is already done!'
            )

        self.step_idx += 1
        graph_action = self._env_context.aidx_to_GraphAction(self.state, action)

        reward = torch.zeros(1)
        step_info_dict = {'mol_is_valid': True}
        self.done = (
            graph_action.action == GraphActionType.Stop or
            self.step_idx == self.max_len
        )

        # If we're done, check that the molecule is valid and compute reward
        if self.done:
            if self.sanitize_samples and not self._env_context.is_sane(self.state):
                step_info_dict['mol_is_valid'] = False
            else:
                reward = self._compute_reward()

        # If we aren't done, apply the action to the current molecule
        else:
            self.state = self._env.step(self.state, graph_action)

            # The action could've still been illegal and added more nodes than
            # allowed. If it is, end the episode and return 0 reward.
            if len(self.state.nodes) > self.max_nodes:
                step_info_dict['mol_is_valid'] = False
                self.done = True

        return (
            self.converter.encode(self._get_embedded_state()),
            reward,
            self.done,
            TURN_IDX,
            step_info_dict
        )


    def _compute_reward(self) -> TensorType[float]:
        mol = self._env_context.graph_to_mol(self.state)
        flat_rewards, is_valid = self.task_obj.compute_flat_rewards([mol])
        return flat_rewards.squeeze()

    def type_name(self) -> str:
        return SEHFragmentMoleculeEnvironment.__name__

    def seed(self, seed: int = None) -> None:
        return

if __name__ == '__main__':
    env = SEHFragmentMoleculeEnvironment()
    import pdb; pdb.set_trace()
    reset_stuff = env.reset()
    step_stuff = env.step((1, 0, 0))
    step_stuff = env.step((0, 0, 0))
    print('hello')
