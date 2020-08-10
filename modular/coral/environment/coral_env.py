import random
from typing import Tuple, List, Optional, Dict, Any, Union

import gym
import numpy as np
import torch
from badger_gym import Agent
from badger_gym.envs.torch_env import TorchEnv, Info, RenderModeNotSupported
from gym.spaces import Box
from prettytable import PrettyTable
from torch import Tensor

from coral.coral_config import CoralConfig
from coral.environment.descriptor_node import Graph
from coral.environment.pygame_rendering import PygameRendering
from coral.environment.robot_segment import RobotSegment
from coral.environment.tasks.task_base import TaskBase

from utils.utils import locate_class


class CoralEnv(TorchEnv):
    """Fully differentiable environment simulating tree-shaped robots."""

    ep_len: int
    segments: List[RobotSegment]
    current_step: int
    fixed_goal: bool
    goal_generated: bool
    goal_hidden: bool

    batch_size: int

    current_obs: Tensor
    last_action: Tensor

    pygame_rendering: Optional[PygameRendering]

    task: TaskBase

    def __init__(self,
                 robot_shape: Union[Graph, List[Graph]],
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
            robot_shape: description of the robot
            ep_len: episode (rollout) length
            randomize: randomize the robot segment lengths?
            seed: reproducibility
            fixed_goal: goal on the fixed position (debug)
            batch_size: run batch_size of environments in parallel (first dimension)
        """
        super().__init__(batch_size, device=device)

        self.seed(seed)

        self.batch_size = batch_size
        self.ep_len = ep_len
        self.current_step = 0

        self.segments = []

        if isinstance(robot_shape, Graph):
            for segment in robot_shape.flatten():
                segment = RobotSegment(angle=segment.angle, length=segment.length, batch_size=batch_size,
                                       device=self.device, rotation_limit=rotation_limit, init_noise=init_noise)
                self.segments.append(segment)
            robot_shape.copy_tree_structure_to(self.segments)
        else:
            # TODO note that only robots with the same topology are supported, topology of the first one is used, while
            # the lengths and angles are taken from each individual robot
            robot_shapes = robot_shape
            assert len(robot_shapes) == self.batch_size
            flat_robots = [robot.flatten() for robot in robot_shapes]

            for segment_id in range(len(flat_robots[0])):
                angles = [robot[segment_id].angle for robot in flat_robots]
                lengths = [robot[segment_id].length for robot in flat_robots]
                segment = RobotSegment(angle=angles, length=lengths, batch_size=batch_size,
                                       device=self.device, rotation_limit=rotation_limit, init_noise=init_noise)
                self.segments.append(segment)
            robot_shapes[0].copy_tree_structure_to(self.segments)

        # TODO randomize
        # if randomize:
        #     self.segments = [Joint(random.random() + 0.1, batch_size, device=self.device) for _ in range(num_joints)]
        # else:
        #     self.segments = [Joint(1, batch_size, device=self.device) for _ in range(num_joints)]

        task_class = locate_class('coral.environment.tasks', task_name)
        self.task = task_class(owner=self, c=c)

        self._observation_space, self._action_space = self._init_spaces()
        self.current_obs = torch.zeros(self.observation_space.shape, device=self.device)
        self.last_action = torch.zeros(self.action_space.shape, device=self.device)

        self.pygame_rendering = None
        self.reset()

    def _init_spaces(self) -> Tuple[gym.Space, gym.Space]:
        observation_space = Box(low=-1., high=1.,
                                shape=(self.batch_size, self.num_segments, self.task.segment_obs_size),
                                dtype=np.float32)
        action_space = Box(low=-1., high=1.,
                           shape=(self.batch_size, self.num_segments, 1),
                           dtype=np.float32)
        return observation_space, action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def is_done(self) -> bool:
        return self.current_step >= self.ep_len

    @property
    def pretty_obs(self) -> PrettyTable:
        """Observation in a human readable form"""
        table = PrettyTable(['JointID'] + self.task.obs_names)

        for segment_id in range(self.num_segments):
            values = [segment_id] + [round(x, 3) for x in self.current_obs[0, segment_id].view(-1).tolist()]
            table.add_row(values)

        table.align = 'l'
        return table

    def _build_obs(self) -> Tensor:
        """Build the observation for this time step"""

        # "forward" through the robot topology
        for segment in self.segments:
            segment.compute_effector_pos()

        # build the observation vector
        return self.task.build_observation()

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.randint(0, 2 ** 32)

        np.random.seed(seed)
        torch.random.manual_seed(seed)

        # TODO task seed?

        # self.action_space.seed(seed)  # TODO might not be deterministic for now!
        # self.observation_space.seed(seed)
        # self.reset()  # TODO should seeding reset?
        return seed

    def reset(self) -> Tensor:
        """ Create target position, reset the Reacher coords, return obs """
        self.current_step = 0

        for segment in self.segments:
            segment.reset()

        self.task.reset()

        if self.pygame_rendering is not None:
            self.pygame_rendering.reset()

        return self._build_obs()

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, bool, Dict]:
        """ Make env step
            Returns:
                Tuple:
                  * observations - Tensor[batch_size, num_segments, JOINT_OBS_SIZE]
                  * reward - Tensor[batch_size, num_segments]
                  * is_done - bool
                  * info - Dictionary[
                     'goal': Tensor[batch_size, 2] "goal position",
                     'eff': Tensor[batch_size, 2] "effector position"]
        """
        self.current_step += 1
        action = action.view(self.action_space.shape)
        self.last_action = action
        for segment_id, segment in enumerate(self.segments):
            segment.make_action(action[:, segment_id])

        self.current_obs = self._build_obs()

        self.task.make_step()

        loss = self.task.compute_loss(self.segments)
        reward = -loss
        return self.current_obs, reward, self.is_done, {'loss': loss.mean()}

    def render(self, mode: str = 'human', policy=None):
        if mode == 'human':
            print(f'Step: {self.current_step} \nCurrent obs:\n{self.pretty_obs}')
            print(f'Last action: {[round(x, 3) for x in self.last_action[0].view(-1).tolist()]}')
            print(f'Loss: {self.task.compute_loss(self.segments).mean().item()}')

            if self.pygame_rendering is None:
                self.pygame_rendering = PygameRendering(self.num_segments, self.task)
            self.pygame_rendering.render(self.segments, self.last_action, policy)
            return

        # elif mode == 'positions':  # TODO support other rendering methods if needed
        #     return self._render_positions()
        # elif mode == 'bokeh':
        #     return self._render_bokeh
        else:
            raise RenderModeNotSupported(f'Unsupported render mode: "{mode}" mode ')

    def to(self, device: str):
        for segment in self.segments:
            segment.to(device)
        self.current_obs.to(device)
        # self.goal_pos.to(device)
        self.last_action.to(device)
        print(f'WARNING: not tested')

    def evaluate(self, agent: Agent) -> Info:
        raise NotImplementedError("To be done")

    def state_dict(self):
        raise NotImplementedError("To be done")

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError("To be done")
