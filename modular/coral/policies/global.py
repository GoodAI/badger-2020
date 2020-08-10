from typing import Optional, Dict, Any, List

import torch
from badger_utils.torch.serializable_module import SerializableModule
from torch import Tensor

from coral.coral_config import CoralConfig
from coral.policies.independent import Independent
from utils.available_device import my_device


class Global(SerializableModule):
    """Global policy that controls everything"""

    expert: SerializableModule
    input_size: int
    output_size: int
    batch_size: int

    current_outputs: Tensor

    def __init__(self,
                 batch_size: int,
                 num_experts: int,
                 input_size_per_expert: int,
                 output_size_per_expert: int,
                 c: CoralConfig):
        super().__init__(device=my_device())

        assert output_size_per_expert == 1, f'Independent policy: output_size_per_expert!=1: {output_size_per_expert}!=1'

        self.output_size = output_size_per_expert * num_experts
        self.input_size = input_size_per_expert * num_experts

        self.input_per_expert = input_size_per_expert
        self.output_per_expert = output_size_per_expert
        self.num_experts = num_experts

        self.batch_size = batch_size
        self.c = c

        self.expert = Independent.make_policy(self.input_size, self.output_size, self.batch_size, c)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = data.view(self.batch_size, self.num_experts * self.input_per_expert)
        output = self.expert.forward(data)
        return output

    def parameters(self, **kwargs) -> List:
        params = []
        params.extend(self.expert.parameters())
        return params

    def reset(self, batch_size: Optional[int] = None):
        self.expert.reset(batch_size)

    def serialize(self) -> Dict[str, Any]:
        return {f'global_policy': self.expert.serialize()}

    def deserialize(self, data: Dict[str, Any]):
        self.expert.deserialize(data[f'global_policy'])
