import math
from typing import List, Tuple, Optional

import pygame
import torch
from torch import Tensor
from torch.nn import Module

from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.pygame_rendering import PygameRendering
from coral.environment.robot_segment import RobotSegment
from coral.environment.tasks.single_point_reacher import SinglePointReacher
from coral.environment.tasks.task_base import TaskBase


class NormalActivation(Module):
    """Gaussian function activation"""

    ca: float
    cb: float
    mean: float
    stdev: float

    def __init__(self, stdev: float = 1.0, mean: float = 0.0):
        super().__init__()
        assert stdev > 0
        self.stdev = stdev
        self.mean = mean
        self.ca = 1 / (self.stdev * math.sqrt(2*math.pi))
        self.cb = 1 / (2 * self.stdev ** 2)

    def forward(self, data: Tensor) -> Tensor:
        a = -(data - self.mean).pow(2) * self.cb
        return self.ca * torch.exp(a)


class Manipulator(TaskBase):
    """Manipulate 1 object from a random position to another randomly chosen position.

    The target position is specified in the observation vector.
    """

    CUSTOM_OBS_NAMES = ['obj_x', 'obj_y', 'gx', 'gy']

    goal_pos: Tensor  # target position
    object_pos: Tensor  # position of the object to be moved around
    fixed_goal: bool
    goal_generated: bool
    loss_func: torch.nn
    owner: CoralEnv
    max_goal_dist: float

    LINE_COLOR = (200, 200, 200)
    OBJECT_COLOR = (100, 100, 100)
    OBJECT_SIZE = 10
    ACT_SCALE = 20
    DIST_MULTIPLIER = 20

    activation: torch.nn
    act_force: str

    def __init__(self,
                 owner: CoralEnv,
                 c: CoralConfig):
        super().__init__(owner)

        self.fixed_goal = c.fixed_goal
        self.goal_generated = False
        self.goal_pos = self._make_zeros(2)
        self.object_pos = self._make_zeros(2)
        self.loss_func = torch.nn.MSELoss(reduction='none')
        self.max_goal_dist = c.max_goal_dist
        self.act_force = c.act_force

        self.activation = NormalActivation(stdev=0.06)

    def custom_obs_per_segment(self, segment_id: int) -> Tensor:
        return torch.cat([self.object_pos, self.goal_pos], dim=1)

    def reset(self):
        # place the object randomly every time
        self.object_pos = SinglePointReacher.make_random_reachable_goal(self.batch_size,
                                                                        self.device,
                                                                        self.max_goal_dist)

        # place the goal either once or every reset
        if not (self.fixed_goal and self.goal_generated):
            self.goal_pos = SinglePointReacher.make_random_reachable_goal(self.batch_size,
                                                                          self.device,
                                                                          self.max_goal_dist)
            self.goal_generated = True

    def _compute_force_from_joint(self, segment: RobotSegment, object_pos: Tensor) -> Tensor:
        # compute the "spring" force scale
        eff_pos = torch.cat([segment.xe, segment.ye], dim=1)
        dist = (object_pos - eff_pos)

        sum_squares = (dist * dist).sum(dim=1)
        total_dist = sum_squares.sqrt()

        # use gaussian function to compute the force scale, then multiply by XY direction
        force_scale = self.activation(total_dist) * self.ACT_SCALE
        force = force_scale.view(-1, 1).expand_as(dist) * dist
        return force

    def _sum_forces(self, object_pos: Tensor) -> Tensor:
        """Given object position, sum forces from all joints"""
        forces = []
        for segment in self.segments:
            force = self._compute_force_from_joint(segment, object_pos)
            forces.append(force)

        # compute the total force
        total_force = torch.stack(forces, dim=2).mean(dim=2)
        return total_force

    def make_step(self):
        """Sum forces from all robot joints, move the object based on the force"""
        self.object_pos += self._sum_forces(self.object_pos)

    def compute_loss(self, segments: List[RobotSegment]) -> Tensor:
        loss = self.loss_func(self.goal_pos, self.object_pos).mean(keepdim=True, dim=1)  # loss per robot
        return loss

    def _draw_object_and_goal(self,
                              renderer: PygameRendering,
                              goal_pos: Tensor,
                              object_pos: Tensor,
                              color: Tuple[int, int, int],
                              obj_color: Tuple[int, int, int]):

        # draw line between the object and the goal
        goal_pos_disp = renderer.to_disp(goal_pos[0, 0].item(), goal_pos[0, 1].item())
        obj_pos = renderer.to_disp(object_pos[0, 0].item(), object_pos[0, 1].item())
        pygame.draw.line(renderer.screen, self.LINE_COLOR, obj_pos, goal_pos_disp, 1)

        # draw the object
        pygame.draw.circle(renderer.screen, obj_color, obj_pos, self.OBJECT_SIZE)

        # draw the goal position
        SinglePointReacher.draw_goal(goal_pos, renderer, color)

    def pygame_render(self, renderer: PygameRendering):
        self._draw_object_and_goal(renderer, self.goal_pos, self.object_pos, self.OBJECT_COLOR, renderer.GOAL_COL)

