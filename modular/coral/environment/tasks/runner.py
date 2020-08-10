import copy
from typing import List, Tuple, Optional

import pygame
import torch
from torch import Tensor


from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.descriptor_node import Graph
from coral.environment.pygame_rendering import PygameRendering
from coral.environment.robot_segment import RobotSegment
from coral.environment.tasks.task_base import TaskBase
import numpy as np


class Runner(TaskBase):
    """Has the ground on top of env.

     Ground can be moved by touching it with the effectors and moving them in the right direction,
     reward is how fast ground moves.
    """

    VEL_SCALE = 1
    MOMENTUM = 0
    # GROUND_DIST = 0.55  # distance of ground from the root anchor point
    GROUND_DIST = 0.7  # distance of ground from the root anchor point
    EFFECTOR_COLOR = (255, 0, 0)

    no_goal: Tensor

    loss_func: torch.nn
    effectors: List[RobotSegment]

    ground_plane: Tensor
    anchor_plane: Tensor

    previous_x_positions: List[Tensor]
    last_total_force: Tensor
    total_distance: Tensor
    velocity: Tensor

    render_line: Optional[List[Tuple[float]]]

    use_pipe: bool = False

    def __init__(self, owner: CoralEnv, c: CoralConfig):
        super().__init__(owner)
        self.no_goal = self._make_zeros(2)
        self.loss_func = torch.nn.MSELoss(reduction='none')
        self.effectors = Graph.get_effectors(self.owner.segments[0])
        self.ground_plane = self._make_ones(1) * self.GROUND_DIST
        self.anchor_plane = self._make_zeros(1)
        self.previous_x_positions = []
        self.last_total_force = self._make_zeros(1)
        self.total_distance = self._make_zeros(1)
        self.velocity = self._make_zeros(1)
        self.render_line = None

        if c.acf == 'leakyrelu':
            self.activation = torch.nn.LeakyReLU()
        elif c.acf == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif c.acf == 'relu':
            self.activation = torch.nn.ReLU()
        elif c.acf == 'tanh':
            self.activation = torch.nn.Tanh()  # the tanh should provide better supervised signal
        elif c.acf == 'gelu':
            # https://pytorch.org/docs/stable/nn.html#gelu
            self.activation = torch.nn.GELU()
        else:
            raise Exception()

    def custom_obs_per_segment(self, segment_id: int):
        # TODO add the distance to the ground to the obs
        return self.no_goal

    def reset(self):
        self.previous_x_positions = []
        self.last_total_force.zero_().detach_()
        self.total_distance.zero_().detach_()
        self.velocity.zero_().detach_()
        self.render_line = None

    def make_step(self):
        """Compute total forces applied to the ground

        - for each joint:
            - compute dist to ground plane
            - apply sigmoid
            - this results in a contact with the plane
            - compute horizontal distance travelled
            - force = contact * distance
        - sum horizontal joint forces
        """

        # first step of rollout = no movement
        if len(self.previous_x_positions) == 0:
            for effector in self.effectors:
                self.previous_x_positions.append(effector.xe.detach())

        forces = []
        DIST_MULTIPLIER = 20

        for effector, prev_pos in zip(self.effectors, self.previous_x_positions):
            # positive distance if the effector is inside ground (because of sigmoid)
            dist = effector.ye - self.ground_plane
            contact = self.activation(DIST_MULTIPLIER * dist)

            if self.use_pipe:
                # penalize also for going under the anchor point
                anchor_dist = - effector.ye
                anchor_contact = self.activation(DIST_MULTIPLIER * anchor_dist)
                contact = contact + anchor_contact

            # TODO if horizontal_dist == 0 and the effector is in the ground, no braking

            horizontal_dist = prev_pos - effector.xe  # movement from right to left is positive
            force = contact * horizontal_dist
            # print(f'contact: {contact[0].item()} * {horizontal_dist[0].item()}= {force[0].item()}')
            forces.append(force)

        all_forces = torch.cat(forces, dim=1)
        # self.last_total_force = torch.sum(all_forces, dim=1, keepdim=True)
        total_force = torch.mean(all_forces, dim=1, keepdim=True)

        self.velocity = self.MOMENTUM * self.velocity + (1 - self.MOMENTUM) * total_force #* 30
        self.total_distance = self.total_distance + self.velocity * self.VEL_SCALE

        # self.total_distance = self.total_distance + self.last_total_force  # TODO ground momentum

        for eff_id, effector in enumerate(self.effectors):
            self.previous_x_positions[eff_id] = effector.xe.detach()

    def compute_loss(self, segments: List[RobotSegment]) -> Tensor:
        return -self.velocity
        # return -self.total_distance

    def _generate_render_line(self):
        RESOLUTION = 0.05
        AMPLITUDE = 0.1
        self.render_line = []

        for step in range(400):
            point = (float(step * RESOLUTION), float(AMPLITUDE * (0.5 - np.random.rand(1))))
            self.render_line.append(point)

    def _render_lines_to(self, renderer: PygameRendering, y_position: float):
        """Render ground visualization to a given y_position on the plane"""
        pygame.draw.line(renderer.screen, (0, 255, 0),  # green ground
                         renderer.to_disp(-1, y_position),
                         renderer.to_disp(1, y_position))

        pos = self.total_distance[0].cpu().item()  # zeroth robot
        prev = self.render_line[0]

        for point in self.render_line[1:]:
            pygame.draw.line(renderer.screen, (0, 255, 0),
                             renderer.to_disp(prev[0] - pos - 1, y_position + prev[1]),
                             renderer.to_disp(point[0] - pos - 1, y_position + point[1]),
                             2)
            prev = point

    def pygame_render(self, renderer: PygameRendering):
        pygame.draw.line(renderer.screen, (0, 0, 255),  # change the color of the root anchor point
                         renderer.to_disp(-1, 0),
                         renderer.to_disp(1, 0))

        if self.render_line is None:
            self._generate_render_line()

        # draw the moving ground
        self._render_lines_to(renderer, self.GROUND_DIST)
        if self.use_pipe:
            self._render_lines_to(renderer, 0)

        for effector in self.effectors:
            pygame.draw.circle(renderer.screen, self.EFFECTOR_COLOR,
                               self._draw_pos(renderer, self._effector_pos(effector)),
                               renderer.R // 2)
