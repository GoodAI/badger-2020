from typing import Tuple, Optional, Dict, Any

import torch
from badger_utils.torch.device_aware import DeviceAwareModule
from torch import Tensor

from coral.coral_config import CoralConfig
from utils.available_device import my_device


class LSTMNet(DeviceAwareModule):

    lstm: torch.nn.LSTM
    out_layer: torch.nn.Linear
    out_activation: torch.nn
    norm: torch.nn.Module

    rand_hidden: bool
    hidden: Tuple[Tensor, Tensor]

    batch_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    input_size: int

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 batch_size: int,
                 c: CoralConfig):
        """
        Args:
            input_size:
            output_size:
            batch_size: How many policies to run in parallel independently (e.g. batch_size * num_experts)
            c:
        """
        super().__init__(device=my_device())

        self.num_layers = c.num_layers
        self.hidden_size = c.hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.input_size = input_size
        self.rand_hidden = c.rand_hidden

        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=c.hidden_size,
                                  num_layers=c.num_layers).to(self.device)
        self.norm = torch.nn.LayerNorm(c.hidden_size, elementwise_affine=False)
        self.out_layer = torch.nn.Linear(in_features=self.hidden_size, out_features=self.output_size).to(self.device)
        self.out_activation = torch.nn.Tanh()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output, self.hidden = self.forward_stateless(data, self.hidden)
        return output

    def forward_stateless(self, data: torch.Tensor, hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        data = data.view(1, -1, self.input_size)
        lstm_out, hidden = self.lstm.forward(data, hidden)
        norm_out = self.norm(lstm_out)  # https://pytorch.org/docs/master/generated/torch.nn.LayerNorm.html
        out = self.out_layer(norm_out)
        outt = self.out_activation(out)
        return outt, hidden

    def reset(self, batch_size: Optional[int] = None):

        batch_size = self.batch_size if batch_size is None else batch_size
        size = (self.num_layers * 1, batch_size, self.hidden_size)

        if self.rand_hidden:
            self.hidden = (torch.rand(size, device=self.device),
                           torch.rand(size, device=self.device))
        else:
            self.hidden = (torch.zeros(size, device=self.device),
                           torch.zeros(size, device=self.device))

    def serialize(self) -> Dict[str, Any]:
        return {'policy': self.state_dict()}

    def deserialize(self, data: Dict[str, Any]):
        self.load_state_dict(data['policy'])

