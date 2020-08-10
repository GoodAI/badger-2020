from collections import namedtuple
from typing import List

import torch
from torch import Tensor

from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.tasks.multi_point_reacher import MultiPointReacher
from coral.environment.tasks.runner import Runner

Point = namedtuple('Point', ['x', 'y'])


class GuideRunner(MultiPointReacher):
    """Curriculum for training the NN to walk for the Runner task"""

    current_step: int
    current_target_id: int
    target_duration: int
    goals_hidden: bool  # goals hidden to the robot?

    STD: float = 0.05  # noise on targets in the batch

    HEIGHT: float = 0.2  # step height
    LENGTH: float = 0.7  # step length
    X_OFFSET: float = 0.1  # offset between the effectors

    # make step template positions around the ground
    template_positions: List[Point] = [
        Point(LENGTH * 0.5, Runner.GROUND_DIST - HEIGHT),  # front above ground
        Point(LENGTH * 0.5, Runner.GROUND_DIST + HEIGHT),
        Point(0, Runner.GROUND_DIST + HEIGHT + 0.1),
        Point(-LENGTH * 0.5, Runner.GROUND_DIST + HEIGHT),
        Point(-LENGTH * 0.5, Runner.GROUND_DIST - HEIGHT),  # rear above ground
        Point(0, Runner.GROUND_DIST - HEIGHT),
    ]

    def __init__(self, owner: CoralEnv, c: CoralConfig):
        super().__init__(owner, c)
        self.target_duration = c.target_duration

        assert self.target_duration <= c.ep_len, 'Target stays too long'
        self.current_target_id = 0
        self.current_step = 0
        self.goals_hidden = c.goals_hidden

    def _get_offset(self, eff_id: int) -> float:
        return -(self.num_effectors - 1) * self.X_OFFSET / 2 + eff_id * self.X_OFFSET

    def _point_to_tensor(self, point: Point, x_offset: float) -> Tensor:
        goal_x = torch.normal(point.x,
                              self.STD,
                              size=(self.batch_size, 1), device=self.device) + x_offset
        goal_y = torch.normal(point.y,
                              self.STD,
                              size=(self.batch_size, 1), device=self.device)
        goal_pos = torch.cat([goal_x, goal_y], dim=1)
        return goal_pos

    def _make_goals(self) -> List[Tensor]:
        current_point = self.template_positions[self.current_target_id]
        # each leg has own x offset
        goals = [
            self._point_to_tensor(current_point, self._get_offset(eff_id))
            for eff_id in range(self.num_effectors)
        ]
        return goals

    def custom_obs_per_segment(self, segment_id: int) -> Tensor:
        if not self.goals_hidden and self.segments[segment_id] in self.effectors:
            eff_id = self.get_effector_id_by_segment_id(segment_id)  # TODO inefficient
            return self.goals[eff_id]

        return self.hidden_goal

    def reset(self):
        self.current_target_id = 0
        self.current_step = 0
        self.goals = self._make_goals()

    def make_step(self):
        super().make_step()

        self.current_step += 1

        if self.current_step > 0 and self.current_step % self.target_duration == 0:
            # next target id
            self.current_target_id += 1
            if self.current_target_id >= len(self.template_positions):
                self.current_target_id = 0

            # build goals
            self.goals = self._make_goals()
