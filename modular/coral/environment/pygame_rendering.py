import random
from typing import Tuple, List

import pygame
import torch
from torch import Tensor

from coral.environment.robot_segment import RobotSegment


# from coral.environment.task import Task


class PygameRendering:
    """Renders the Coral robot in a pygame window. Renders just the zeroth robot in the batch."""

    DISP_HEIGHT = 500
    DISP_WIDTH = 500

    COL = (0, 100, 255)
    EFF_COL = (0, 0, 255)
    GOAL_COL = (255, 0, 0)
    R = 10

    num_segments: int
    episode_actions: List

    task: 'Task'

    def __init__(self, num_segments: int, task: 'Task'):
        pygame.init()
        self.num_segments = num_segments
        self.screen = pygame.display.set_mode((self.DISP_WIDTH, self.DISP_HEIGHT))
        self.colors = [self.random_color() for _ in range(self.num_segments)]
        self.episode_actions = []
        self.task = task

    def reset(self):
        # TODO not clearing this draws actions from multiple rollouts, crop graph reasonably is used
        self.episode_actions.clear()  # rendering

    def to_disp(self, x: float, y: float) -> Tuple[int, int]:
        """ Convert to display coordinate frame """
        w_half = self.DISP_WIDTH // 2
        h_half = self.DISP_HEIGHT // 2
        smaller_half = min(w_half, h_half)
        return int(x * 0.95 * smaller_half + w_half), int(h_half - (y * 0.95 * smaller_half))

    def tens_to_disp(self, px: Tensor, py: Tensor) -> Tuple[int, int]:
        px = px[0].to('cpu').item()
        py = py[0].to('cpu').item()
        return self.to_disp(x=px, y=py)

    @staticmethod
    def random_color() -> Tuple[int, int, int]:
        return tuple([random.randint(0, 255) for _ in range(3)])

    def _draw_actions(self, last_action: Tensor):
        """ Draws the graph of actions beneath the robot """

        # store the last actions for the moving graph
        self.episode_actions.append(last_action[0].view(-1).to('cpu').tolist())

        TIME_RES = 3
        RIGHT_EDGE = self.DISP_WIDTH - TIME_RES - int(self.DISP_WIDTH * 0.2)
        ZERO = 3 * self.DISP_HEIGHT / 4
        SCALE = self.DISP_HEIGHT / 4 - int(self.DISP_HEIGHT * 0.02)
        LW = 1

        # TODO this can actually draw outside the screen (but does not crash)

        # background
        graph_left_top = (RIGHT_EDGE - len(self.episode_actions) * TIME_RES, ZERO - SCALE)
        graph_width_height = (len(self.episode_actions) * TIME_RES, SCALE * 2)
        pygame.draw.rect(self.screen, (245, 245, 245), pygame.Rect(graph_left_top, graph_width_height))
        # zero line
        pygame.draw.line(self.screen, (0, 0, 0),
                         (RIGHT_EDGE, ZERO),
                         (RIGHT_EDGE - len(self.episode_actions) * TIME_RES, ZERO), LW)

        # lines
        pos = RIGHT_EDGE
        if len(self.episode_actions) > 1:
            for action_id, color in enumerate(self.colors):
                # extract one line & draw it sequentially
                line = [step[action_id] for step in self.episode_actions]
                prev_value = line[-1]
                for step, value in enumerate(reversed(line[:-1])):
                    start = (pos - step * TIME_RES, ZERO - prev_value * SCALE)
                    end = (pos - step * TIME_RES - TIME_RES, ZERO - value * SCALE)
                    pygame.draw.line(self.screen, color, start, end, LW)
                    prev_value = value

    def draw_image(self,
                   target: Tensor,
                   upper_left: Tuple[float, float],
                   width: float) -> Tuple[int, int]:
        """ Draw a given tensor to the coordinate frame with origin at center of the image, axes go right and top

        Args:
            target: 2D tensor to draw
            upper_left: position of the upper left corner of the image
            width: total width of the image (in the coordinate frame <-1,1>

        Returns: width, height of the image (in the <-1, 1> coordinate frame)
        """
        SPACE_SIZE = 1  # space between pixels (in screen pixels)

        # compute the pixel size and space size based on the width and num_cols in the tensor
        assert len(target.shape) == 2
        num_cols = target.shape[0]
        pixel_size = width / num_cols  # pixel size in <-1, 1> coords
        pixel_size = int(pixel_size / 2 * self.DISP_WIDTH)  # pixel size in screen pixels

        x_shift, y_shift = self.to_disp(upper_left[0], upper_left[1])  # shift in screen coords

        for line_id, line in enumerate(target):
            for column_id, pixel in enumerate(line):

                goal_rectangle = pygame.Rect(
                    column_id * (pixel_size + SPACE_SIZE) + x_shift,
                    line_id * (pixel_size + SPACE_SIZE) + y_shift,
                    pixel_size,  # width
                    pixel_size)  # height

                # color
                pixel_val = min(max(pixel.item(), -1), 1)  # sanitization
                pixel_val = pixel_val * 255
                color = (-pixel_val, 0, 0) if pixel_val < 0 else (0, pixel_val, 0)

                # draw
                pygame.draw.rect(self.screen, color, goal_rectangle)

        # TODO this conversion is not very accurate
        w = ((num_cols + 1) * (pixel_size + SPACE_SIZE)) / self.DISP_WIDTH * 2
        h = ((target.shape[1] + 1) * (pixel_size + SPACE_SIZE)) / self.DISP_HEIGHT * 2

        return w, h

    def render(self,
               segments: List[RobotSegment],
               last_action: Tensor,
               policy=None):
        """Draw the robot and actions to the pygame window"""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(f'Kill the render process instead')
                # self.current_step = self.ep_len  # artificially end the ep

        # draw the game
        self.screen.fill((255, 255, 255))

        # draw history of robot's actions
        self._draw_actions(last_action)

        # grass
        pygame.draw.line(self.screen, (0, 255, 0), self.to_disp(-1, 0), self.to_disp(1, 0))

        # draw the zeroth robot in the batch
        for segment, color in zip(segments, self.colors):
            # pygame.draw.circle(self.screen, color,
            #                    self._tens_to_disp(segment.xo[0], segment.yo[0]),
            #                    self.R, 1)
            pygame.draw.line(self.screen, color,
                             self.tens_to_disp(segment.xo[0], segment.yo[0]),
                             self.tens_to_disp(segment.xe[0], segment.ye[0]), 5)

        # task specific
        # goal_rectangle = pygame.Rect(*self.to_disp(goal_pos[0, 0].item(), goal_pos[0, 1].item()),
        #                              self.R, self.R)
        # pygame.draw.rect(self.screen, self.GOAL_COL, goal_rectangle)
        #
        # draw the end effector
        # pygame.draw.circle(self.screen, self.EFF_COL,
        #                    self._tens_to_disp(segments[-1].xe[0], segments[-1].ye[0]),
        #                    self.R // 2)
        self.task.pygame_render(self)

        if policy is not None:
            # TODO a hacky way how to render part of the policy in the env simulator:(
            # print(f'rendering policy comm mask')

            # if the matrix is available, draw the matrix (realtime discovery) otherwise draw the mask
            mask = None
            if hasattr(policy, 'comm_matrix'):
                mask = policy.comm_matrix[0]  # [batch_size, num_segments, num_segments]
            elif hasattr(policy, 'comm_mask'):
                mask = policy.comm_mask[0, :, :, 0]  # [batch_size, num_segments, num_segments, hidden_size]
            if mask is not None:
                m = torch.max(mask)  # max autoscale (assume positive values)
                mask = mask / (m + 0.01)

                width = 0.5
                pos = (-0.9, -0.9 + width)  # start where to draw
                _, _ = self.draw_image(mask, upper_left=pos, width=width)

        pygame.display.flip()
