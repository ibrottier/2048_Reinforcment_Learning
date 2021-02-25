import logging
import datetime as dt

import ray
from ray import tune
import ray.rllib.agents.impala as impala
from Environment_Class import TwentyFortyEightEnvironment

stop_iters = 30000
stop_reward = 1000000
stop_timesteps = 10000000000

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    ray.init(dashboard_host="0.0.0.0")

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        "episode_reward_mean": stop_reward,
    }

    env = TwentyFortyEightEnvironment()

    config = impala.DEFAULT_CONFIG.copy()

    config["env"] = TwentyFortyEightEnvironment
    config["num_workers"] = 3
    config["num_gpus"] = 0
    config["framework"] = "tfe"
    config["log_level"] = "DEBUG"
    config["rollout_fragment_length"] = 1000
    config["train_batch_size"] = config["rollout_fragment_length"] * 15
    print(config)

    results = tune.run("IMPALA", stop=stop, config=config, checkpoint_at_end=True, checkpoint_freq=100, local_dir="./models/")

    ray.shutdown()