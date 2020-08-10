import math
from abc import ABC
from collections import namedtuple
from typing import List, Optional, Tuple
from typing import TypeVar, Generic

from torch import Tensor
import numpy as np

Origin = namedtuple('Origin', ['x', 'y', 'r'])

TNode = TypeVar('TNode')


class NodeBase(Generic[TNode], ABC):

    children: List[TNode]
    parent: Optional[TNode]
    global_origin: Origin

    def __init__(self):
        self.children = []
        self.parent = None

    def add_child(self, child: TNode):
        if child in self.children:
            raise Exception('cycles in the graph not supported')
        self.children.append(child)
        child.parent = self


class DescriptorNode(NodeBase['Node']):
    """A Node with purpose of describing ths shape of the robot"""

    angle: float  # rotation from the origin, radians
    length: float

    xe: float
    ye: float
    re: float

    global_origin: Origin

    def __init__(self,
                 angle: float = 0.0,
                 length: float = 0.0,
                 global_origin: Origin = Origin(x=0, y=0, r=math.pi * 0.5)):
        """ Create a robot Node, which supports both float and tensor operations
        Args:
            angle: angle of the controllable joint
            length: length of the segment
            global_origin: optional global position of robot's origin (just for the root node)
        """
        super().__init__()

        self.angle = angle
        self.length = length
        self.global_origin = global_origin

    @property
    def xo(self) -> float:
        return self._parent_effector()[0]

    @property
    def yo(self) -> float:
        return self._parent_effector()[1]

    @property
    def ro(self) -> float:
        return self._parent_effector()[2]

    def _parent_effector(self) -> Tuple:
        if self.parent is not None:
            return self.parent.xe, self.parent.ye, self.parent.re
        return self.global_origin.x, self.global_origin.y, self.global_origin.r

    def apply_joint(self):
        """Read the Parent's effector (or default origin) and apply my transformation,
         save my effector position"""

        self.re = self.ro + self.angle
        self.xe = self.xo + math.cos(self.re) * self.length
        self.ye = self.yo + math.sin(self.re) * self.length


class Graph:
    root: Optional[DescriptorNode]

    def __init__(self):
        self.root = None

    @property
    def size(self) -> int:
        return len(self.flatten())

    @staticmethod
    def get_children(node: DescriptorNode) -> List[DescriptorNode]:
        if len(node.children) == 0:
            return [node]
        res = []
        for off in node.children:
            res.extend(Graph.get_children(off))

        return [node] + res

    def flatten(self) -> List[DescriptorNode]:
        return self.get_children(self.root)

    def compute_robot(self):
        for joint in self.flatten():
            joint.apply_joint()

    def _longest_path(self, node: DescriptorNode) -> float:
        """Find the longest path from the root to the edge"""
        if len(node.children) == 0:
            if isinstance(node.length, Tensor):
                return float(node.length[0].cpu().item())
            return node.length

        longest = 0
        for child in node.children:
            path_len = self._longest_path(child)
            longest = max(longest, path_len)
        if isinstance(node.length, Tensor):
            return longest + float(node.length[0].cpu().item())
        return longest + node.length

    def normalize(self):
        """Normalize the robot size to have the longest path from root to effector = 1"""
        max_len = self._longest_path(self.root)
        for joint in self.flatten():
            joint.length = round(joint.length / max_len, 3)

    def copy_tree_structure_to(self, target_nodes: List[NodeBase]):
        """Assumed use: this Graph is used as a descriptor of the robot structure.
         This can be copied to another graph, to do this:
         -for each node in flattened list create corresponding node of any type
         -call this to copy parent/offspring links to the new nodes

         note: target_node should implement the method Node.add_offspring(target: Node)
         """
        nodes, offspring_ids = self._flatten_with_ids()
        assert len(nodes) == len(target_nodes)

        for node, target_node, off_ids in zip(nodes, target_nodes, offspring_ids):
            for off_id in off_ids:
                target_node.add_child(target_nodes[off_id])

    def _flatten_with_ids(self) -> Tuple[List[DescriptorNode], List[List[int]]]:
        nodes = self.flatten()
        result = []

        for node in nodes:
            children_ids = []

            for child in node.children:
                for pos, target in enumerate(nodes):
                    if target == child:
                        children_ids.append(pos)

            result.append(children_ids)
        return nodes, result

    @staticmethod
    def get_effectors(node: TNode) -> List[TNode]:
        """Given the root node, return list of effectors (Nodes that do not have any children)."""
        if len(node.children) == 0:
            return [node]
        res = []
        for child in node.children:
            res.extend(Graph.get_effectors(child))

        return res

    def get_connection_matrix(self) -> np.ndarray:
        """Get the 2D matrix defining the robot structure"""
        result = np.zeros((self.size, self.size))
        nodes, all_offspring_ids = self._flatten_with_ids()  # for each node, get a list of its offspring ids

        for node_id, offspring_ids in enumerate(all_offspring_ids):
            for off_id in offspring_ids:
                result[node_id, off_id] = 1.0

        return result


