import copy
import math
import time
from typing import Tuple, List

from coral.coral_config import CoralConfig
from coral.environment.descriptor_node import Graph, DescriptorNode
import numpy as np

from coral.fractal_parser.fractal import generate_program
from coral.fractal_parser.fractal_to_graph import program_to_graph


def random_angle(angle_range: float) -> float:
    # angle_range = math.pi / 2
    return np.random.uniform(low=-angle_range, high=angle_range)


def random_len(min_segment_len: float) -> float:
    return np.random.uniform(low=min_segment_len, high=1.0)


class RobotGenerator:

    def __init__(self, num_segments: int, p_branching: float, min_segment_len: float, angle_range: float):
        """Generate random robot topology with the following properties:
            -random constant rotations of joints
            -fixed number of segments
            -branching optional

        Args:
            num_segments: fixed num segments of the robot (num experts)
            p_branching: probability that new segment will be connected to randomly chosen existing expert
            min_segment_len: minimum size of segment (max is 1)
        """

        self.num_segments = num_segments
        self.p_branching = p_branching
        self.min_segment_length = min_segment_len
        self.angle_range = angle_range

        assert num_segments > 0
        assert 0 <= p_branching <= 1
        assert 0 < min_segment_len <= 1
        assert 0 <= angle_range <= 3.14

    def _random_node(self) -> DescriptorNode:
        return DescriptorNode(angle=random_angle(self.angle_range), length=random_len(self.min_segment_length))

    def next_robot(self) -> Graph:

        np.random.seed(None)

        graph = Graph()
        last_node = self._random_node()
        graph.root = last_node

        while graph.size < self.num_segments:
            new_node = self._random_node()

            if np.random.uniform() < self.p_branching:
                # branch
                parent_id = np.random.choice(graph.size)
                parent = graph.flatten()[parent_id]
                parent.add_child(new_node)
            else:
                # append to the end
                last_node.add_child(new_node)

            last_node = new_node

        graph.normalize()
        return graph

    def next_robot_batch(self, batch_size: int) -> List[Graph]:
        """Return batch_size of robots with the same topology, but different shapes"""
        robot = self.next_robot()
        robots = [copy.deepcopy(robot) for _ in range(batch_size)]
        for robot in robots:
            for segment in robot.flatten():
                segment.angle = random_angle(self.angle_range)
                segment.length = random_len(self.min_segment_length)

        for robot in robots:
            robot.normalize()

        return robots


def get_program(type: str) -> Tuple[List[str], List[str], str]:
    """Used for selecting the fractal programs to generate the robot shape"""
    if type == 'a':
        ruleInput = ['F', 'X']
        ruleOutput = ["FF", "F-[[X]+X]+F[+FX]-X"]
        start = "X"
    elif type == 'b':
        ruleInput = ['F']
        ruleOutput = ['F[-F][+F]']
        start = "F"
    elif type == 'linear':
        ruleInput = ['F']
        ruleOutput = ['FF']
        start = 'F'
    elif type == 'fork':  # a nice manipulator - designed for 1 iteration
        ruleInput = ['F']
        ruleOutput = ['FFF[-FF][+FF]']
        start = 'F'
    else:
        raise Exception('unsupported program type')

    return ruleInput, ruleOutput, start


def make_robot(c: CoralConfig):
    if c.type == 'generated':
        generator = RobotGenerator(c.num_segments, c.p_branching, c.min_segment_length)
        return generator.next_robot()
    else:
        ruleInput, ruleOutput, start = get_program(c.type)
        graph = program_to_graph(generate_program(c.iterations, start, ruleInput, ruleOutput), c.batch_size)
        return graph


