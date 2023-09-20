# from crystal_design.envs.seh_frag_molecule_env import SEHFragmentMoleculeEnvironment
# from crystal_design.envs.crystal_env import CrystalEnv
from hive.envs.base import BaseEnv, ParallelEnv
from hive.envs.env_spec import EnvSpec
from hive.envs.gym_env import GymEnv
from hive.utils.registry import registry
import sys
sys.path.append('/network/scratch/p/prashant.govindarajan/crystal_design_project/code/crystal-design/') 
   
try:
    from crystal_design.envs.crystal_env import CrystalEnv
    print('Imported Successfully')
except ImportError:
    CrystalEnv = None
    print('Failed to Import!')

try:
    from crystal_design.envs.crystal_env import CrystalEnvV2
    print('Imported Successfully')
except ImportError:
    CrystalEnvV2 = None
    print('Failed to Import!')


try:
    from crystal_design.envs.crystal_env import CrystalEnvV3
    print('Imported Successfully')
except ImportError:
    CrystalEnvV3 = None
    print('Failed to Import!')

try:
    from crystal_design.envs.crystal_env import CrystalGraphEnvPerov
    print('Imported Successfully')
except ImportError:
    CrystalGraphEnvPerov = None
    print('Failed to Import!')

registry.register_all(
    BaseEnv,
    {
        "GymEnv": GymEnv,
        "CrystalEnv": CrystalEnv,
        "CrystalEnvV2": CrystalEnvV2,
        "CrystalEnvV3": CrystalEnvV3,
        "CrystalGraphEnvPerov": CrystalGraphEnvPerov,
    },
)

get_env = getattr(registry, f"get_{BaseEnv.type_name()}")
