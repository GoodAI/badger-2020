from typing import Optional, Dict, Any, Tuple

import gym
from badger_gym import Agent
from badger_gym.envs.torch_env import TorchEnv, Info, TTorchEnv
from torch import Tensor

from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.descriptor_node import Graph
from coral.robot_generators import RobotGenerator


class CoralEnvWrapper(TorchEnv):
    """Support different robot shapes between resets (make new env between resets)"""

    my_env: CoralEnv
    generator: RobotGenerator
    graph: Graph

    def __init__(self,
                 robot_generator: RobotGenerator,
                 c: CoralConfig,
                 ep_len: int = 100,
                 task_name: str = 'SinglePointReacher',
                 randomize: bool = False,
                 seed: Optional[int] = None,
                 fixed_goal: bool = False,
                 batch_size: int = 1,
                 init_noise: float = 0.0,
                 device: Optional[str] = None,
                 rotation_limit: Optional[float] = None):
        """
        Args:
            robot_generator: a class that generates now robots
            ep_len: episode (rollout) length
            randomize: randomize the robot segment lengths?
            seed: reproducibility
            fixed_goal: goal on the fixed position (debug)
            batch_size: run batch_size of environments in parallel (first dimension)
        """
        super().__init__(batch_size, device=device)

        self.c = c
        self.ep_len = ep_len
        self.task_name = task_name
        self.randomize = randomize
        self.seed = seed
        self.fixed_goal = fixed_goal
        self.init_noise = init_noise
        self.rotation_limit = rotation_limit

        self.generator = robot_generator

        self.my_env = self._make_env()

    @property
    def num_segments(self) -> int:
        return self.my_env.num_segments

    def reset(self) -> Tensor:
        self.my_env = self._make_env()
        return self.my_env.reset()

    def _make_env(self) -> CoralEnv:
        robot_batch = self.generator.next_robot_batch(self.batch_size)

        self.graph = robot_batch[0]

        env = CoralEnv(
            robot_batch,
            self.c,
            self.ep_len,
            self.task_name,
            self.randomize,
            None,  # TODO hast to be random seed?
            self.fixed_goal,
            self.batch_size,
            self.init_noise,
            self.device,
            self.rotation_limit)
        return env

    @property
    def observation_space(self) -> gym.Space:
        return self.my_env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.my_env.action_space

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, bool, Info]:
        return self.my_env.step(action)

    def render(self, mode='human', policy=None):
        return self.my_env.render(mode, policy=policy)

    def state_dict(self):
        return self.my_env.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.my_env.load_state_dict(state_dict)

    def to(self: TTorchEnv, device: str) -> TTorchEnv:
        return self.my_env.to(device)

    def evaluate(self, agent: Agent) -> Info:
        return self.my_env.evaluate(agent)
