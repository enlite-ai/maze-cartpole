"""Smoke tests to ensure standard configurations can be run."""
import glob
from typing import Dict

import pytest

from maze.test.shared_test_utils.run_maze_utils import run_maze_job
from maze.utils.timeout import Timeout

# Configurations to be tested
configurations = [
    ["conf_train", {"algorithm": "ppo",
                    "env": "cartpole_env"}],
    ["conf_train", {"algorithm": "ppo", "model": "cartpole_template_model",
                    "env": "cartpole_env"}],
    ["conf_train", {"algorithm": "ppo", "model": "cartpole_custom_model", "critic": "cartpole_custom_state_critic",
                    "env": "cartpole_env"}],
    ["conf_train", {"algorithm": "ppo", "wrappers": "cartpole_wrappers",
                    "env": "cartpole_env"}],

    ["conf_rollout", {"env": "cartpole_env"}],
    ["conf_rollout", {"policy": "cartpole_heuristic_policy", "env": "cartpole_env"}],
    ["conf_rollout", {"policy": "cartpole_heuristic_policy", "runner": "sequential",
                      "runner.record_trajectory": True, "runner.render": True, "runner.n_episodes": 1, "env": "cartpole_env"}],

    ["conf_train", {"+experiment": "cartpole_hard_ppo"}],
]


@pytest.mark.parametrize("config_name,hydra_overrides", configurations)
def test_standard_configurations(config_name: str, hydra_overrides: Dict[str, str]):
    # run training
    try:
        with Timeout(seconds=15):
            run_maze_job(hydra_overrides, config_module="maze.conf", config_name=config_name)
    except TimeoutError:
        # ignore timeout errors, we don't wait for the training to end
        pass

    if config_name == "conf_train":
        # load tensorboard log
        tf_summary_files = glob.glob("*events.out.tfevents*")
        assert len(tf_summary_files) == 1, f"expected exactly 1 tensorflow summary file {tf_summary_files}"
