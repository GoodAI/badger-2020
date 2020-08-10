from typing import Optional, Dict, Any

import torch
from badger_utils.torch.device_aware import DeviceAwareModule
from badger_utils.torch.serializable_module import SerializableModule
from torch import Tensor

from coral.coral_config import CoralConfig
from coral.policies.independent import Independent
from utils.available_device import my_device


class Shared(DeviceAwareModule):
    """Experts share weights, no communication, one dim per expert output.

    The policy here handles the remapping: [batch_size] vs [batch_size * num_experts]
    """

    expert: SerializableModule
    input_per_expert: int

    current_outputs: Tensor

    def __init__(self,
                 batch_size: int,
                 num_experts: int,
                 input_size_per_expert: int,
                 output_size_per_expert: int,
                 c: CoralConfig):
        super().__init__(device=my_device())

        assert output_size_per_expert == 1, f'Shared policy: output_size_per_expert!=1: {output_size_per_expert}!=1'

        self.num_experts = num_experts
        self.batch_size = batch_size
        self.output_size = output_size_per_expert * num_experts
        self.input_size = input_size_per_expert * num_experts

        self.input_per_expert = input_size_per_expert
        self.c = c

        self.expert = Independent.make_policy(self.input_per_expert, output_size_per_expert, batch_size, c)
        self.current_outputs = torch.zeros(self.output_size)

        self.reset()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = data.view(self.num_experts * self.batch_size, self.input_per_expert)

        output = self.expert.forward(data)
        self.current_outputs = output.view(-1)

        assert self.current_outputs.numel() == self.output_size * self.batch_size
        return self.current_outputs

    def parameters(self, **kwargs):
        return self.expert.parameters()

    def reset(self, batch_size: Optional[int] = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        self.expert.reset(self.num_experts * batch_size)

    def serialize(self) -> Dict[str, Any]:
        return self.expert.serialize()

    def deserialize(self, data: Dict[str, Any]):
        self.expert.deserialize(data)

