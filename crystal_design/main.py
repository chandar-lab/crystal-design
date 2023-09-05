# from hive.runners.single_agent_loop import SingleAgentRunner
# from hive.runners.single_agent_loop import main
import sys
# sys.path.append('/network/scratch/p/prashant.govindarajan/crystal_design_project/code/crystal-design/') 
from crystal_design.envs.crystal_env import CrystalGraphEnvMP
from crystal_design.runner import Runner

if __name__ == "__main__":
    
    data_path = '/home/pragov/scratch/online/crystal-design/3696_val.pkl'
    runner = Runner(data_path = data_path)
    runner.train_agent()
