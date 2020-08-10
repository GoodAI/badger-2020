import math
from typing import List, Optional, Tuple

import pygame
import torch
from torch import Tensor

from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.pygame_rendering import PygameRendering
from coral.environment.robot_segment import RobotSegment
from coral.environment.tasks.task_base import TaskBase


class SinglePointReacher(TaskBase):
    """Reach to a randomly selected point in a 2D space."""

    goal_pos: Tensor
    fixed_goal: bool
    goal_generated: bool
    loss_func: torch.nn
    owner: CoralEnv

    def __init__(self,
                 owner: CoralEnv,
                 c: CoralConfig):
        super().__init__(owner)
        self.fixed_goal = c.fixed_goal
        self.goal_generated = False
        self.goal_pos = torch.zeros((self.batch_size, 2), dtype=torch.float, device=self.device)
        self.loss_func = torch.nn.MSELoss(reduction='none')

    def custom_obs_per_segment(self, segment_id: int) -> Tensor:
        return self.goal_pos

    def reset(self):
        if not (self.fixed_goal and self.goal_generated):
            self.goal_pos = self.make_random_reachable_goal(self.batch_size, self.device)
            self.goal_generated = True

    def make_step(self):
        pass

    @staticmethod
    def make_random_reachable_goal(batch_size: int, device: str, max_dist: float = 1.0) -> Tensor:
        """Make 1 (per robot in batch) random 2D position in a circle of unit length radius."""
        goal_pos = torch.zeros((batch_size, 2), device=device)
        origin_dists = torch.rand(batch_size) * max_dist
        angles = torch.rand(batch_size) * math.pi
        goal_pos[:, 0] = angles.cos() * origin_dists
        goal_pos[:, 1] = angles.sin() * origin_dists
        return goal_pos

    def _effector_pos(self) -> Tensor:
        """Position of effector of the last segment in the list"""
        return torch.cat([self.segments[-1].xe, self.segments[-1].ye], dim=1)

    def compute_loss(self, segments: List[RobotSegment]) -> Tensor:
        eff_pos = self._effector_pos()
        loss = self.loss_func(self.goal_pos, eff_pos).mean(keepdim=True, dim=1)  # loss per robot
        return loss

    @staticmethod
    def draw_goal(goal_pos: Tensor, renderer: PygameRendering, color: Optional[Tuple[int, int, int]] = None):
        color = color if color is not None else renderer.GOAL_COL

        goal_pos_disp = renderer.to_disp(goal_pos[0, 0].item(), goal_pos[0, 1].item())
        goal_pos_disp = [goal_pos_disp[0] - renderer.R // 2, goal_pos_disp[1] - renderer.R // 2]

        goal_rectangle = pygame.Rect(*goal_pos_disp, renderer.R, renderer.R)
        pygame.draw.rect(renderer.screen, color, goal_rectangle)

    def pygame_render(self, renderer: PygameRendering):
        self.draw_goal(self.goal_pos, renderer)

        # draw the end effector
        eff_pos = self._effector_pos()
        pygame.draw.circle(renderer.screen, renderer.EFF_COL,
                           renderer.to_disp(eff_pos[0, 0].item(), eff_pos[0, 1].item()),
                           renderer.R // 2)
