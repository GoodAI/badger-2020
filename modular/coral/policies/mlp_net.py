from typing import Optional, Dict, Any

import torch
from badger_utils.torch.serializable_module import SerializableModule
from torch import nn

from coral.coral_config import CoralConfig
from utils.available_device import my_device


class MLPNet(SerializableModule):

    layers: nn.Sequential
    input_size: int

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 batch_size: int,
                 c: CoralConfig):
        """Does not care about batch_size"""
        super().__init__(device=my_device())

        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, c.hidden_size),
            nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),  # TODO add num_layers here
            # nn.LeakyReLU(),
            nn.Linear(c.hidden_size, output_size),
            nn.Tanh()
        ).to(self.device)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = data.view(-1, self.input_size)
        out = self.layers(data)
        return out

    def reset(self, batch_size: Optional[int] = None):
        pass

    def serialize(self) -> Dict[str, Any]:
        return {'policy': self.state_dict()}

    def deserialize(self, data: Dict[str, Any]):
        self.load_state_dict(data['policy'])
