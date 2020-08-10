from typing import Tuple, Optional, Dict, Any, List

import torch
from badger_utils.torch.serializable_module import SerializableModule
from torch import Tensor

from coral.coral_config import CoralConfig
from utils.available_device import my_device
from utils.utils import locate_class


class Independent(SerializableModule):
    """Independent policies without communication"""

    experts: List[SerializableModule]
    input_per_expert: int
    output_per_expert: int

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
        self.batch_size = batch_size
        self.num_experts = num_experts
        self.c = c

        self.experts = [
            self.make_policy(self.input_per_expert, self.output_per_expert, batch_size, c)
            for _ in range(num_experts)
        ]
        self.current_outputs = torch.zeros(self.output_size)

    @staticmethod
    def make_policy(input_size: int, output_size: int, batch_size: int, c: CoralConfig) -> SerializableModule:
        policy_class = locate_class('reacher.policies', c.subpolicy)
        policy = policy_class(input_size, output_size, batch_size, c)
        return policy

    def _get_input(self, expert_id: int, data: Tensor) -> Tensor:
        start = expert_id * self.input_per_expert
        return data[start:start + self.input_per_expert]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Split the data per expert, run through each policy independently,
         stitch back to the [batch_size, num_experts, output_per_expert] size and return
        """
        data = data.view(self.batch_size, self.num_experts, self.input_per_expert)
        outputs = []

        for expert_id, expert in enumerate(self.experts):
            expert_input = data[:, expert_id]
            outputs.append(
                expert.forward(expert_input).view(self.batch_size, 1, self.output_per_expert)
            )

        # stack by the first dimension (see above) and reshape appropriately
        self.current_outputs = torch.cat(outputs, dim=1).view(self.batch_size * self.num_experts, self.output_per_expert)
        return self.current_outputs

    def parameters(self, **kwargs) -> List:
        params = []
        for expert in self.experts:
            params.extend(expert.parameters())
        return params

    def reset(self, batch_size: Optional[int] = None):
        for expert in self.experts:
            expert.reset(batch_size)

    def serialize(self) -> Dict[str, Any]:
        params = {}
        for expert_id, expert in enumerate(self.experts):
            params[f'expert_{expert_id}'] = expert.serialize()

        return params

    def deserialize(self, data: Dict[str, Any]):
        for expert_id, expert in enumerate(self.experts):
            expert.deserialize(data[f'expert_{expert_id}'])
