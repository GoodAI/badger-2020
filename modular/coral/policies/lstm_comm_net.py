from typing import Dict, Any, Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from coral.policies.comm_net_base import CommNetBase
from coral.coral_config import CoralConfig


class LstmCommNet(CommNetBase):
    """ Implementation of the recurrent version of CommNet"""

    rand_hidden: bool
    num_layers: int

    f_net: nn.LSTM
    hidden: Tuple[Tensor, Tensor]
    last_comm: Tensor

    f_net_norm: nn.LayerNorm

    def __init__(self,
                 batch_size: int,
                 num_experts: int,
                 input_size_per_expert: int,
                 output_size_per_expert: int,
                 conf: CoralConfig):

        super().__init__(batch_size, num_experts, input_size_per_expert, output_size_per_expert, conf)

        self.rand_hidden = conf.rand_hidden
        self.num_layers = conf.num_layers

        if not conf.skip_connections:
            print(f'WARNING: LstmCommNet: will use skip_connections')

        """ Networks """
        self.f_net = nn.LSTM(input_size=2 * self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        """ Normalization """
        self.f_net_norm = nn.LayerNorm(self.hidden_size)

        self.reset()
        self.to(self.device)

    def deserialize(self, data: Dict[str, Any]):
        super().deserialize(data)
        self.f_net.load_state_dict(data['f_net'])
        self.f_net_norm.load_state_dict(data['f_net_norm'])

    def serialize(self) -> Dict[str, Any]:
        result = super().serialize()
        result['f_net'] = self.f_net.state_dict()
        result['f_net_norm'] = self.f_net_norm.state_dict()
        return result

    def forward(self, observation: Tensor) -> Tensor:
        """Process the input, communicate once and return the action"""
        obs_processed = self._process_observation(observation)

        for step in range(self.num_passes):
            f_inputs = torch.cat([self.last_comm, obs_processed], dim=1).unsqueeze(0)

            # compute new hidden
            f_net_out, self.hidden = self.f_net.forward(f_inputs, self.hidden)
            hidden_normed = self.f_net_norm(f_net_out)

            # communicate
            if self.comm_enabled:
                self.last_comm = self._communicate(hidden_normed, self.comm_mask)

        return self._compute_action(hidden_normed)

    def reset(self, batch_size: Optional[int] = None):

        batch_size = self.batch_size if batch_size is None else batch_size
        size = (self.num_layers, batch_size * self.num_experts, self.hidden_size)

        if self.rand_hidden:
            self.hidden = (torch.rand(size, device=self.device),
                           torch.rand(size, device=self.device))
        else:
            self.hidden = (torch.zeros(size, device=self.device),
                           torch.zeros(size, device=self.device))

        self.last_comm = torch.zeros(size[1:], device=self.device)
