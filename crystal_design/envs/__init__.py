from crystal_design.envs.seh_frag_molecule_env import SEHFragmentMoleculeEnvironment
# from crystal_design.envs.crystal_env import CrystalEnv
from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.gym_env import GymEnv
from hive.utils.registry import registry
    
try:
    from crystal_design.envs.crystal_env import CrystalEnv
except ImportError:
    CrystalEnv = None

registry.register_all(
    BaseEnv,
    {
        "GymEnv": GymEnv,
        "CrystalEnv": CrystalEnv
    },
)

get_env = getattr(registry, f"get_{BaseEnv.type_name()}")
