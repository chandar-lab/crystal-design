import argparse
import copy

from hive import agents as agent_lib
from hive import envs
from hive.runners.base import Runner
from hive.runners.utils import TransitionInfo, load_config
from hive.utils import experiment, loggers, schedule, utils
from hive.utils.registry import get_parsed_args
from hive.runners.single_agent_loop import SingleAgentRunner


class SingleAgentRunnerOffline(SingleAgentRunner):
    def __init__(
        self,
        environment,
        agent,
        logger,
        experiment_manager,
        train_steps,
        test_frequency,
        test_episodes,
        stack_size,
        max_steps_per_episode=27000,
        data_path = ''
    ):
        super(SingleAgentRunnerOffline).__init__(
                                            environment,
                                            agent,
                                            logger,
                                            experiment_manager,
                                            train_steps,
                                            test_frequency,
                                            test_episodes,
                                            stack_size,
                                            max_steps_per_episode
                                        )
        self.data_path = data_path
    
    def run_episode(self):
        pass