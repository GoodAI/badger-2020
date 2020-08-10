from typing import List, Tuple, Optional

import torch
from torch import Tensor

from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.pygame_rendering import PygameRendering
from coral.environment.robot_segment import RobotSegment
from coral.environment.tasks.manipulator import Manipulator
from coral.environment.tasks.single_point_reacher import SinglePointReacher


class MultiPointManipulator(Manipulator):
    """Manipulator with multiple objects + goals"""

    CUSTOM_OBS_NAMES: List[str]
    OBS_PER_GOAL = ['obj_x', 'obj_y', 'gx', 'gy']

    goal_pos: List[Tensor]  # target positions
    object_pos: List[Tensor]  # positions of the objects to be moved around

    num_points: int  # num objects

    goal_colors: Optional[List[Tuple[int, int, int]]]  # their colors for rendering

    def __init__(self, owner: CoralEnv, c: CoralConfig):
        super().__init__(owner, c)

        self.num_points = c.num_points
        assert self.num_points > 0

        self.CUSTOM_OBS_NAMES = []
        for point_id in range(self.num_points):
            names = [f'{point_id}_{obs}' for obs in self.OBS_PER_GOAL]
            self.CUSTOM_OBS_NAMES.extend(names)

        self.goal_colors = None

    def custom_obs_per_segment(self, segment_id: int) -> Tensor:
        """Each segment sees all objects and all corresponding goals"""
        data = []
        for obj_pos, goal_pos in zip(self.object_pos, self.goal_pos):
            data.extend([obj_pos, goal_pos])
        return torch.cat(data, dim=1)

    def reset(self):
        # random object positions
        self.object_pos = [
            SinglePointReacher.make_random_reachable_goal(self.batch_size, self.device, self.max_goal_dist)
            for _ in range(self.num_points)
        ]

        # place the goal positions either once or every reset
        if not (self.fixed_goal and self.goal_generated):
            self.goal_pos = [
                SinglePointReacher.make_random_reachable_goal(self.batch_size, self.device, self.max_goal_dist)
                for _ in range(self.num_points)
            ]
            self.goal_generated = True

    def make_step(self):
        """Move each object based on the sum of forces from robot joints"""

        # TODO optimization here
        for object_pos in self.object_pos:
            object_pos += self._sum_forces(object_pos)

    def compute_loss(self, segments: List[RobotSegment]) -> Tensor:
        positions = torch.cat(self.object_pos, dim=1)
        goals = torch.cat(self.goal_pos, dim=1)
        return self.loss_func(goals, positions).mean(keepdim=True, dim=1)  # loss per robot [batch_size, 1]

    def pygame_render(self, renderer: PygameRendering):

        if self.goal_colors is None:
            self.goal_colors = []
            for _ in range(self.num_points):
                self.goal_colors.append(PygameRendering.random_color())

        for object_pos, goal_pos, color in zip(self.object_pos, self.goal_pos, self.goal_colors):
            self._draw_object_and_goal(renderer, goal_pos, object_pos, color, color)
