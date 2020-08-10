import copy
import math
import time
from collections import namedtuple
from typing import List, Tuple

import pygame

from coral.environment.descriptor_node import DescriptorNode, Graph
from coral.fractal_parser.showrobot import _rand_color, to_disp

DISP_HEIGHT = 500
DISP_WIDTH = 500

COL = (0, 100, 255)
EFF_COL = (0, 0, 255)
GOAL_COL = (255, 0, 0)

R = 7


Pos = namedtuple('Pos', ['x', 'y'])


class Edge:
    """Just an edge from start to end in 2D"""

    def __init__(self, start: Pos, end: Pos):
        self.start = start
        self.end = end

    def equal_to(self, other: 'Edge') -> bool:
        if self.start == other.start and self.end == other.end:
            return True
        if self.end == other.start and self.start == other.end:
            return True
        return False

    def tostr(self):
        return f'({self.start.x},{self.start.y}->{self.end.x},{self.end.y})'

    @property
    def angle(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        if dx == 0:
            return math.pi * 0.5
        return math.atan2(dy, dx)

    @property
    def length(self) -> float:
        return math.sqrt((self.start.x - self.end.x) ** 2 + (self.start.y - self.end.y) ** 2)


class MyTurtle:
    """My turtle which generates list of edges on 2D"""

    edges: List[Edge]
    current_pos: Pos
    current_angle: float
    precision: int

    def __init__(self, precision: int = 25):
        self.current_pos = Pos(0, 0)
        self.current_angle = math.pi / 2
        self.edges = []
        self.precision = precision

    @property
    def heading(self) -> float:
        return math.degrees(self.current_angle)

    @property
    def pos(self) -> Pos:
        return copy.deepcopy(self.current_pos)

    def forward(self, front: float):
        x = round(self.current_pos.x + math.cos(self.current_angle) * front, self.precision)
        y = round(self.current_pos.y + math.sin(self.current_angle) * front, self.precision)
        end_pos = Pos(x, y)
        self.edges.append(Edge(self.current_pos, end_pos))
        self.current_pos = copy.deepcopy(end_pos)

    def left(self, turn):
        self.current_angle += math.radians(-turn)

    def right(self, turn):
        self.current_angle += math.radians(turn)

    def setpos(self, pos: Pos):
        assert isinstance(pos, Pos)
        self.current_pos = Pos(pos.x, pos.y)

    def setheading(self, angle: float):
        self.current_angle = math.radians(angle)

    @staticmethod
    def _is_in(edge: Edge, edges: List[Edge]):
        for ed in edges:
            if edge.equal_to(ed):
                return True
        return False

    def unique_edges(self) -> List[Edge]:
        result = []
        for edge in self.edges:
            if not self._is_in(edge, result):
                result.append(edge)
        return result


def program_to_lines(program, front, turn) -> List[Edge]:
    """Given the program, run the turtle which generates edges, return list of unique edges"""
    stack = []
    dirstack = []

    mt = MyTurtle()

    for x in program:
        if x == 'F':
            # turtle.forward(front)
            mt.forward(front)
        elif x == '-':
            # turtle.left(turn)
            mt.left(turn)
        elif x == '+':
            # turtle.right(turn)
            mt.right(turn)
        elif x == '[':  # push
            # stack.append(turtle.pos())
            # dirstack.append(turtle.heading())
            stack.append(mt.pos)
            dirstack.append(mt.heading)
        elif x == ']':  # pop
            # turtle.penup()
            post = stack.pop()
            direc = dirstack.pop()
            mt.setpos(post)
            mt.setheading(direc)
            # turtle.setpos(post)
            # turtle.setheading(direc)
            # turtle.pendown()
        # else:
        #     print(f'X')  # end effector

    return mt.unique_edges()


def get_offsprings(edge: Edge, edges: List[Edge]) -> Tuple[List[Edge], List[Edge]]:
    """Given the edge, split the list of edges to offspring edges and others"""
    offsprings = []
    others = []
    if edge in edges:
        edges.remove(edge)

    for target in edges:
        if edge.end == target.start:
            offsprings.append(target)
        else:
            others.append(target)

    return offsprings, others


def find_offsprings(parent_edge: Edge,
                    parent_node: DescriptorNode,
                    unprocessed_edges: List[Edge]) -> List[Edge]:
    """Compares parent_edge's end with starts of other unprocessed edges,
     if the points match, add as an offspring to the graph"""
    offsprings, not_consumed = get_offsprings(parent_edge, unprocessed_edges)
    if len(offsprings) == 0:
        return not_consumed
    for offspring in offsprings:
        angle = parent_edge.angle - offspring.angle
        off_node = DescriptorNode(length=offspring.length, angle=angle)
        parent_node.add_child(off_node)
        not_consumed = find_offsprings(offspring, off_node, not_consumed)

    return not_consumed


def edges_to_graph(edges: List[Edge]) -> Graph:
    """Converts list of edges to graph, assumes root to be at the beginning of the list"""
    graph = Graph()
    edge_current = edges[0]
    node_current = DescriptorNode(angle=edge_current.angle, length=edge_current.length)
    graph.root = node_current

    unprocessed = find_offsprings(edge_current, node_current, edges[1:])

    assert len(unprocessed) == 0
    assert graph.size == len(edges)
    graph.root.angle = 0  # origin is artificially rotated pi/2, compensate for this here
    return graph


def program_to_graph(program: str, front=1, turn=33) -> Graph:
    print(f'Generating edges...')
    edges = program_to_lines(program, front, turn)
    print(f'Generating graph...')
    graph = edges_to_graph(edges)
    graph.normalize()
    print(f'Done, generated graph of size: {graph.size}')
    return graph


def draw_edges(edges: List[Edge], blocking=True):
    """Just a debug, draws edges to plane to check consistency with the original fractal"""
    pygame.init()
    screen = pygame.display.set_mode((DISP_WIDTH, DISP_HEIGHT))
    colors = [_rand_color() for _ in range(len(edges))]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print(f'should quit')

    # draw the game
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 255, 0), to_disp(-1, 0), to_disp(1, 0))

    # compute all potential robots, draw the first one
    for edge, color in zip(edges, colors):
        # xo, yo, ro = joint.parent_effector()  # origin of the segment

        # pygame.draw.circle(screen, color, to_disp(edge.start.x, edge.start.y), R, 1)
        pygame.draw.line(screen,
                         color,
                         to_disp(edge.start.x, edge.start.y),
                         to_disp(edge.end.x, edge.end.y), 1)

    # TODO support for more effectors
    # draw the end effector
    # pygame.draw.circle(screen, EFF_COL, to_disp(joints[-1].xe, joints[-1].ye), R // 2)
    pygame.display.flip()
    if not blocking:
        return
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(f'should quit')
        time.sleep(0.5)


