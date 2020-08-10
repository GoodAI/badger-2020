import math
import random
import time
from typing import Tuple

import pygame

from coral.environment.descriptor_node import Graph, DescriptorNode

DISP_HEIGHT = 500
DISP_WIDTH = 500

COL = (0, 100, 255)
EFF_COL = (0, 0, 255)
GOAL_COL = (255, 0, 0)

R = 5
THICKNESS = 2


def to_disp(x: float, y: float) -> Tuple[int, int]:
    """ Convert to display coordinate frame """
    w_half = DISP_WIDTH // 2
    h_half = DISP_HEIGHT // 2
    smaller_half = min(w_half, h_half)
    return int(x * 0.95 * smaller_half + w_half), int(h_half - (y * 0.95 * smaller_half))


def _rand_color() -> Tuple[any, ...]:
    return tuple([random.randint(0, 255) for _ in range(3)])


def show_robot(graph: Graph, blocking: bool = True):
    pygame.init()
    screen = pygame.display.set_mode((DISP_WIDTH, DISP_HEIGHT))
    colors = [_rand_color() for _ in range(graph.size)]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print(f'should quit')

    # draw the game
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 255, 0), to_disp(-1, 0), to_disp(1, 0))

    # origin and joints
    joints = graph.flatten()
    [joint.apply_joint() for joint in joints]  # compute the robot

    # compute all potential robots, draw the first one
    for joint, color in zip(joints, colors):

        # pygame.draw.circle(screen, color, to_disp(xo, yo), R, 1)
        pygame.draw.line(screen, color,
                         to_disp(joint.xo, joint.yo),
                         to_disp(joint.xe, joint.ye), THICKNESS)

    # TODO support for more effectors
    # draw the end effector
    pygame.draw.circle(screen, EFF_COL, to_disp(joints[-1].xe, joints[-1].ye), R // 2)
    pygame.display.flip()

    if not blocking:
        return
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(f'should quit')
        time.sleep(0.5)


def move_robot(graph: Graph):
    pygame.init()
    screen = pygame.display.set_mode((DISP_WIDTH, DISP_HEIGHT))
    colors = [_rand_color() for _ in range(graph.size)]
    joints = graph.flatten()

    while True:
        for joint in joints:
            joint.angle += (0.5 - random.random()) * 0.01

        [joint.apply_joint() for joint in joints]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(f'should quit')

        # draw the game
        screen.fill((255, 255, 255))
        pygame.draw.line(screen, (0, 255, 0), to_disp(-1, 0), to_disp(1, 0))

        # origin and joints
        joints = graph.flatten()
        [joint.apply_joint() for joint in joints]  # compute the robot

        # compute all potential robots, draw the first one
        for joint, color in zip(joints, colors):

            # pygame.draw.circle(screen, color, to_disp(xo, yo), R, 1)
            pygame.draw.line(screen, color,
                             to_disp(joint.xo, joint.yo),
                             to_disp(joint.xe, joint.ye), THICKNESS)

        # TODO support for more effectors
        # draw the end effector
        pygame.draw.circle(screen, EFF_COL, to_disp(joints[-1].xe, joints[-1].ye), R // 2)
        pygame.display.flip()


if __name__ == '__main__':
    graph = Graph()
    root = DescriptorNode(angle=0, length=0.5)
    graph.root = root
    a = DescriptorNode(angle=math.pi * 0.2, length=0.2)
    b = DescriptorNode(angle=math.pi * (-0.2), length=0.4)
    c = DescriptorNode(angle=math.pi * 0.3, length=0.1)
    d = DescriptorNode(angle=math.pi * 0.3, length=0.1)
    e = DescriptorNode(angle=math.pi * (-0.3), length=0.1)
    f = DescriptorNode(angle=math.pi * 0.4, length=0.4)
    root.add_child(a)
    root.add_child(b)
    b.add_child(c)
    a.add_child(d)
    a.add_child(e)
    e.add_child(f)

    show_robot(graph)

    time.sleep(5)
