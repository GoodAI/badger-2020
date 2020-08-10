from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import Tensor

from coral.environment.pygame_rendering import PygameRendering
from coral.environment.robot_segment import RobotSegment


class TaskBase(ABC):
    """Base for defining tasks in the Coral environment"""

    owner: 'CoralEnv'

    # custom things added by the tasks
    CUSTOM_OBS_NAMES = ['gx', 'gy']
    # mostly default names (should be common for most of the tasks)
    DEFAULT_OBS_NAMES = ['ox', 'oy', 'sin(o)', 'cos(o)', 'ex', 'ey', 'sin(angle)', 'cos(angle)']

    def __init__(self, owner: 'CoralEnv'):
        self.owner = owner

    @property
    def segment_obs_size(self) -> int:
        """
        Observation size per one segment.
        This can vary between tasks, change OBS_NAMES and get_obs_per_segment() or TODO to do this.
        """
        return len(self.CUSTOM_OBS_NAMES + self.DEFAULT_OBS_NAMES)

    @property
    def obs_names(self) -> List[str]:
        return self.CUSTOM_OBS_NAMES + self.DEFAULT_OBS_NAMES

    @property
    def batch_size(self) -> int:
        return self.owner.batch_size

    @property
    def device(self) -> str:
        return self.owner.device

    @property
    def num_segments(self) -> int:
        return len(self.owner.segments)

    @property
    def segments(self) -> List[RobotSegment]:
        return self.owner.segments

    # TODO seed, to_device?

    @abstractmethod
    def reset(self):
        """Reset the task"""
        pass

    def build_observation(self) -> Tensor:
        """Build the observation for each segment, assemble them into a given shape"""

        # prepare the data of shape [batch_size, num_segments, task.segment_obs_size]
        current_obs = torch.zeros(self.owner.observation_space.shape, device=self.device)

        for segment_id, segment in enumerate(self.segments):

            obs_per_segment = torch.cat(
                [
                    self.custom_obs_per_segment(segment_id),
                    segment.xo, segment.yo,
                    segment.ro.sin(), segment.ro.cos(),
                    segment.xe, segment.ye,
                    segment.angles.sin(), segment.angles.cos()
                ], dim=1)

            current_obs[:, segment_id] = obs_per_segment

        return current_obs

    @abstractmethod
    def custom_obs_per_segment(self, segment_id: int) -> Tensor:
        """Define and return a custom observation data per each segment"""
        pass

    @abstractmethod
    def make_step(self):
        pass

    @abstractmethod
    def compute_loss(self, segments: List[RobotSegment]) -> Tensor:
        """Loss per-robot"""
        pass

    @abstractmethod
    def pygame_render(self, renderer: PygameRendering):
        """Render anything task-specific nex to the robot in pygame window"""
        pass

    def _make_zeros(self, size: int = 1):
        return torch.zeros((self.batch_size, size), device=self.device)

    def _make_ones(self, size: int = 1):
        return torch.ones((self.batch_size, size), device=self.device)

    @staticmethod
    def _effector_pos(effector: RobotSegment) -> Tensor:
        """Position of given effector"""
        return torch.cat([effector.xe, effector.ye], dim=1)

    @staticmethod
    def _draw_pos(renderer: PygameRendering, pos: Tensor) -> Tuple[float, float]:
        """Draw the position to the pygame window"""
        return renderer.to_disp(pos[0, 0].item(), pos[0, 1].item())
