import math
from typing import Optional, Tuple, List, Union

import torch
from torch import Tensor

from coral.environment.descriptor_node import NodeBase, Origin


class RobotSegment(NodeBase['RobotSegment']):

    batch_size: int
    ACTION_SCALE = 0.05

    angles: Tensor
    lengths: Tensor

    xe: Tensor
    ye: Tensor
    re: Tensor

    device: str
    rotation_limit: Optional[float]
    init_noise: float

    def __init__(self,
                 angle: Union[float, List[float]],
                 length: Union[float, List[float]],
                 batch_size: int,
                 device: Optional[str] = None,
                 rotation_limit: Optional[float] = None,
                 init_noise: float = 0.0):
        """Each joint rotates and translates. Compared to Reacher, this supports tree structured robots"""
        super().__init__()
        self.device = self._get_device(device)

        self.batch_size = batch_size
        self.rotation_limit = rotation_limit
        self.init_noise = init_noise

        # angle and length can be common for batch or unique per item in batch (different robots in the batch)
        if isinstance(angle, float) or isinstance(angle, int):
            assert isinstance(length, float) or isinstance(length, int)
            self.base_angles = self._make_ones() * angle
            self.lengths = self._make_ones() * length
        else:
            # TODO not tested and might not be used
            assert isinstance(angle, List) and len(angle) == self.batch_size
            assert isinstance(length, List) and len(length) == self.batch_size
            angle: List = angle
            length: List = length
            self.base_angles = torch.Tensor(angle).view(self.batch_size, -1).to(self.device)
            self.lengths = torch.Tensor(length).view(self.batch_size, -1).to(self.device)

        # this creates the tensor Origin, which is used in case no parent node found
        self.global_origin = Origin(self._make_zeros(),
                                    self._make_zeros(),
                                    torch.ones((batch_size, 1), device=self.device) * math.pi * 0.5)
        self.xe = self._make_zeros()
        self.ye = self._make_zeros()
        self.re = self._make_zeros()
        self.angles = self._make_zeros()
        self.reset()

    @staticmethod
    def _get_device(device: Optional[str] = None):
        if device is None:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _make_zeros(self, size: int = 1):
        return torch.zeros((self.batch_size, size), device=self.device)

    def _make_ones(self, size: int = 1):
        return torch.ones((self.batch_size, size), device=self.device)

    def reset(self):
        """Reset the segment to nearly initial position"""
        self.lengths = self.lengths.detach()
        self.base_angles = self.base_angles.detach()
        # centered noise to the common_angle
        # self.angles = torch.rand((self.batch_size, 1), device=self.device)
        # self.angles = (-self.angles + 0.5) * 0.1 * 2 * math.pi

        if self.init_noise <= 0:
            self.init_noise = 0.001
        self.angles = torch.normal(mean=0.0, std=self.init_noise, size=(self.batch_size, 1), device=self.device)
        # at -pi/pi the effect of actions is 0 (if rotation_limit=0.5)
        self.angles.clamp_(min=-math.pi * 0.8, max=math.pi * 0.8)

        self.compute_effector_pos()

    def make_action(self, action: Tensor):
        """Accept action from the policy (rotation in range <-1,1>)"""
        assert action.shape == self.angles.shape
        if self.rotation_limit is None:
            self.angles = self.angles + action * self.ACTION_SCALE
        else:
            # rotation limit 1 means range [-pi/2, pi/2]
            self.angles = self.angles + self.ACTION_SCALE * action * (self.angles * self.rotation_limit).cos()

    def to(self, device: str):
        self.lengths.to(device)
        self.angles.to(device)
        self.xe.to(device)
        self.ye.to(device)
        self.re.to(device)

    @property
    def xo(self) -> Tensor:
        return self._parent_effector()[0]

    @property
    def yo(self) -> Tensor:
        return self._parent_effector()[1]

    @property
    def ro(self) -> Tensor:
        return self._parent_effector()[2]

    def _parent_effector(self) -> Tuple:
        """Get position & rotation of the previous robot segment (or origin, if parent not found)"""
        if self.parent is not None:
            return self.parent.xe, self.parent.ye, self.parent.re
        return self.global_origin.x, self.global_origin.y, self.global_origin.r

    def compute_effector_pos(self):
        """Compute position of my effector based on the parent effector and my rotation"""
        self.re = self.ro + self.angles + self.base_angles
        self.xe = self.xo + self.re.cos() * self.lengths
        self.ye = self.yo + self.re.sin() * self.lengths

