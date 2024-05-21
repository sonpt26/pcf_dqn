from network import NetworkEnv
import os
import logging
import logging.config
import yaml
import time

os.environ["KERAS_BACKEND"] = "tensorflow"
with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)
logger = logging.getLogger("my_logger")

env = NetworkEnv()
observation = env.reset()
while True:
    action = env.action_space.sample()
    # action = [1, 0.25, 0.5]
    logger.info("action %s", action)
    observation, reward, done, _ = env.step(action)
    logger.info("observation %s", observation)
    env.reset()
    time.sleep(1)
    print("$$$$$$$$$$$$$$$$$$$$$")
