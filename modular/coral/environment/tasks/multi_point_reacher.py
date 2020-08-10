import math
from typing import List, Tuple, Optional

import pygame
import torch
from torch import Tensor

from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.descriptor_node import Graph
from coral.environment.pygame_rendering import PygameRendering
from coral.environment.robot_segment import RobotSegment
from coral.environment.tasks.single_point_reacher import SinglePointReacher
from coral.environment.tasks.task_base import TaskBase

import numpy as np


class MultiPointReacher(TaskBase):
    """For each effector of the robot, select a goal position.

    In this case, only the effectors see their respective goals (communication needed).
    """
    fixed_goal: bool
    goals_generated: bool
    loss_func: torch.nn

    effectors: List[RobotSegment]
    goals: List[Tensor]  # goal per effector
    hidden_goal: Tensor
    goal_colors: Optional[List[Tuple]]

    show_directions: bool = True  # show line between each effector and his goal?

    def __init__(self,
                 owner: CoralEnv,
                 c: CoralConfig):
        super().__init__(owner)
        self.fixed_goal = c.fixed_goal
        self.goals_generated = False
        self.loss_func = torch.nn.MSELoss(reduction='none')

        self.effectors = Graph.get_effectors(self.owner.segments[0])
        self.goals = [
            torch.zeros((self.batch_size, 2), dtype=torch.float, device=self.device)
            for _ in range(self.num_effectors)
        ]
        self.hidden_goal = torch.zeros((self.batch_size, 2), dtype=torch.float, device=self.device)
        self.goal_colors = None
        print(f'Task: num_effectors = num_goals = {self.num_effectors}')

    @property
    def num_effectors(self) -> int:
        return len(self.effectors)

    def get_effector_id_by_segment_id(self, segment_id: int) -> int:
        segment = self.segments[segment_id]
        for eff_id, eff in enumerate(self.effectors):
            if eff == segment:
                return eff_id
        raise Exception('Effector not found!')

    def custom_obs_per_segment(self, segment_id: int) -> Tensor:
        if self.segments[segment_id] in self.effectors:
            eff_id = self.get_effector_id_by_segment_id(segment_id)  # TODO inefficient
            return self.goals[eff_id]

        return self.hidden_goal

    def _make_random_goal(self) -> Tensor:
        """Note: some goals will not be reachable by some effectors"""
        return SinglePointReacher.make_random_reachable_goal(self.batch_size, self.device)

    def reset(self):
        if not (self.fixed_goal and self.goals_generated):
            self.goals = [self._make_random_goal() for _ in range(self.num_effectors)]
            self.goals_generated = True

        self.step = 0
        self.changed = -1
        self.cooldown = 50

    step: int = 0
    changed: int = -1
    dir: np.ndarray
    speed: float = 0.05
    cooldown: int

    def make_step(self):
        MOVE_GOAL = False  # after an initial period it moves one goal, waits several steps and moves another..

        if MOVE_GOAL:
            while self.cooldown > 0:
                self.cooldown -= 1
                return

            self.step += 1
            if self.step % 50 == 0:
                self.changed += 1
                self.cooldown = 25

                self.dir = self.speed * (np.random.random(size=2) - 0.5)

                if self.changed >= self.num_effectors:
                    self.changed = 0

            if self.changed != -1:
                print(f'moving goal {self.changed} by {self.dir}')
                g = self.goals[self.changed]
                g[0][0] = g[0][0] + self.dir[0]
                g[0][1] = g[0][1] + self.dir[1]

    def compute_loss(self, segments: List[RobotSegment]) -> Tensor:
        losses = []
        for effector, goal in zip(self.effectors, self.goals):
            eff_pos = self._effector_pos(effector)
            # TODO: loss could be computed more efficiently,
            #  but the dim is kept so that it can be used for reward per agent
            loss = self.loss_func(goal, eff_pos).mean(keepdim=True, dim=1)
            losses.append(loss)

        # cat the losses per effector, mean them per robot, return [batch_size, 1] of losses
        loss = torch.cat(losses, dim=1).mean(keepdim=True, dim=1)
        return loss

    def pygame_render(self, renderer: PygameRendering):
        # generate random colors and save them
        if self.goal_colors is None:
            self.goal_colors = []
            for _ in self.effectors:
                self.goal_colors.append(PygameRendering.random_color())

        # draw effector and goal with the same color
        for color, effector, goal in zip(self.goal_colors, self.effectors, self.goals):

            # draw the goal
            goal_rectangle = pygame.Rect(*self._draw_pos(renderer, goal),
                                         renderer.R, renderer.R)
            pygame.draw.rect(renderer.screen, color, goal_rectangle)

            # draw the end effector
            eff_pos = self._effector_pos(effector)
            pygame.draw.circle(renderer.screen, color,
                               self._draw_pos(renderer, eff_pos),
                               renderer.R // 2)

            if self.show_directions:
                pygame.draw.line(renderer.screen, color,
                                 self._draw_pos(renderer, goal),
                                 self._draw_pos(renderer, eff_pos), 1)
