from typing import Dict, Any

import torch
from torch import Tensor
from torch import nn

from coral.policies.comm_net_base import CommNetBase
from coral.coral_config import CoralConfig


class RnnCommNet(CommNetBase):
    """ Implementation of the CommNet, feed forward version with shared f^i networks per comm. step """

    skip_connections: bool

    f_net: nn.Linear
    f_net_norm: nn.LayerNorm

    def __init__(self,
                 batch_size: int,
                 num_experts: int,
                 input_size_per_expert: int,
                 output_size_per_expert: int,
                 conf: CoralConfig):

        super().__init__(batch_size, num_experts, input_size_per_expert, output_size_per_expert, conf)

        self.skip_connections = conf.skip_connections
        f_net_inp_size = 3 * self.hidden_size if self.skip_connections else 2 * self.hidden_size

        """ Networks """
        self.f_net = nn.Linear(f_net_inp_size, self.hidden_size)  # TODO multi-layer MLP here?

        """ Normalization """
        self.f_net_norm = nn.LayerNorm(self.hidden_size)

        self.to(self.device)

    def deserialize(self, data: Dict[str, Any]):
        super().deserialize(data)
        self.f_net.load_state_dict(data[f'f_net'])
        self.f_net_norm.load_state_dict(data[f'f_net_norm'])

    def serialize(self) -> Dict[str, Any]:
        result = super().serialize()
        result[f'f_net'] = self.f_net.state_dict()
        result[f'f_net_norm'] = self.f_net_norm.state_dict()
        return result

    def forward(self, observation: Tensor) -> Tensor:
        """Here:
            -the f^i is shared across the time steps
            -hidden state is initialized from the observation each env. step.
        """
        obs_processed = self._process_observation(observation)

        hidden_zero = obs_processed
        hidden = hidden_zero  # initial hidden state is computed from the input
        comm_input = self._zeros()  # initial comm input

        """Do K communication passes"""
        for pass_id in range(self.num_passes):

            # build the input to the f^t
            f_inputs = [comm_input, hidden, hidden_zero] if self.skip_connections else [comm_input, hidden]
            f_inputs = torch.cat(f_inputs, dim=1)

            # compute new hidden
            hidden = self.f_net_norm(self.activation(self.f_net.forward(f_inputs)))

            # communicate
            if self.comm_enabled and pass_id < (self.num_passes - 1):
                comm_input = self._communicate(hidden, self.comm_mask)

        return self._compute_action(hidden)


