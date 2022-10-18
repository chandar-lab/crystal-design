from hive.runners.single_agent_loop import SingleAgentRunner
from hive import agents as agent_lib
from hive import envs
from hive.runners.base import Runner
from hive.runners.utils import TransitionInfo, load_config
from hive.utils import experiment, loggers, schedule, utils
from hive.utils.registry import get_parsed_args
import argparse


def set_up_experiment(config):
    """Returns a :py:class:`SingleAgentRunner` object based on the config and any
    command line arguments.
    Args:
        config: Configuration for experiment.
    """

    args = get_parsed_args(
        {
            "seed": int,
            "train_steps": int,
            "test_frequency": int,
            "test_episodes": int,
            "max_steps_per_episode": int,
            "stack_size": int,
            "resume": bool,
            "run_name": str,
            "save_dir": str,
        }
    )
    config.update(args)
    full_config = utils.Chomp(copy.deepcopy(config))

    if "seed" in config:
        utils.seeder.set_global_seed(config["seed"])

    environment_fn, full_config["environment"] = envs.get_env(
        config["environment"], "environment"
    )
    environment = environment_fn()
    env_spec = environment.env_spec

    # Set up loggers
    logger_config = config.get("loggers", {"name": "NullLogger"})
    if logger_config is None or len(logger_config) == 0:
        logger_config = {"name": "NullLogger"}
    if isinstance(logger_config, list):
        logger_config = {
            "name": "CompositeLogger",
            "kwargs": {"logger_list": logger_config},
        }

    logger_fn, full_config["loggers"] = loggers.get_logger(logger_config, "loggers")
    logger = logger_fn()

    agent_fn, full_config["agent"] = agent_lib.get_agent(config["agent"], "agent")
    agent = agent_fn(
        observation_space=env_spec.observation_space[0],
        action_space=env_spec.action_space[0],
        stack_size=config.get("stack_size", 1),
        logger=logger,
    )

    # Set up experiment manager
    saving_schedule_fn, full_config["saving_schedule"] = schedule.get_schedule(
        config["saving_schedule"], "saving_schedule"
    )
    experiment_manager = experiment.Experiment(
        config["run_name"], config["save_dir"], saving_schedule_fn()
    )
    experiment_manager.register_experiment(
        config=full_config,
        logger=logger,
        agents=agent,
    )
    # Set up runner
    runner = SingleAgentRunner(
        environment,
        agent,
        logger,
        experiment_manager,
        config.get("train_steps", -1),
        config.get("test_frequency", -1),
        config.get("test_episodes", 1),
        config.get("stack_size", 1),
        config.get("max_steps_per_episode", 1e9),
    )
    if config.get("resume", False):
        runner.resume()

    return runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-p", "--preset-config")
    parser.add_argument("-a", "--agent-config")
    parser.add_argument("-e", "--env-config")
    parser.add_argument("-l", "--logger-config")
    args, _ = parser.parse_known_args()
    if args.config is None and args.preset_config is None:
        raise ValueError("Config needs to be provided")
    config = load_config(
        args.config,
        args.preset_config,
        args.agent_config,
        args.env_config,
        args.logger_config,
    )
    runner = set_up_experiment(config)
    runner.run_training()


if __name__ == "__main__":
    main()
