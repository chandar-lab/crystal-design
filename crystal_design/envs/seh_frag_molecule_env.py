from hive.envs.base import BaseEnv
from crystal_design.utils import get_device
from gflownet.tasks import SEHTask
from typing import Tuple
from torchtyping import TensorType

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

        self.device = get_device()

    def reset(self) -> Tuple[OBSERVATION_TYPE, int]:
        self.state = self._env.new()
        self.done = False

        return self._get_embedded_state(), TURN_IDX

    def _get_embedded_state(self) -> TensorType[float]:
        singleton_torch_graphs = [self._env_context.graph_to_Data(self.state)]

        return self._env_context.collate(
            singleton_torch_graphs
        ).to(device=self.device).squeeze()

    def step(self, action: int) -> Tuple[
        OBSERVATION_TYPE,
        float, # Reward
        bool,  # Done
        int,   # Turn (irrelevant for this env)
        dict   # Custom information dict
    ]:
        if not isinstance(action, int):
            raise TypeError(
                'Action must be of type int, was of type %s' % type(action)
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
            graph_action == GraphAction.Stop or self.step_idx == self.max_len
        )

        # If we're done, check that the molecule is valid and compute reward
        if self.done:
            if self.sanitize_samples and not self._env_context.is_sane(self.state):
                step_info_dict['mol_is_valid'] = False
            else:
                reward = self._compute_reward()

        # If we aren't done, apply the action to the current molecule
        else:
            gp = self._env.step(self.state, graph_action)

            # The action could've still been illegal and added more nodes than
            # allowed. If it is, end the episode and return 0 reward.
            if len(gp.nodes) > self.max_nodes:
                step_info_dict['mol_is_valid'] = False
                self.done = True

        return (
            self._get_embedded_state(),
            reward,
            self.done,
            TURN_IDX,
            step_info_dict
        )


    def _compute_reward(self) -> TensorType[float]:
        flat_rewards, is_valid = self.task_obj.compute_flat_rewards([self.state])
        return flat_rewards

    def type_name(self) -> str:
        return SEHFragmentMoleculeEnvironment.__name__
