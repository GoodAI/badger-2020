from typing import Dict, Any
from typing import List

import torch
from torch import Tensor
from torch import nn

from coral.policies.comm_net_base import CommNetBase
from coral.coral_config import CoralConfig


class FfCommNet(CommNetBase):
    """ Implementation of the CommNet, feed forward version with different f^i networks per comm. step """

    skip_connections: bool

    f_nets: List[nn.Linear]
    f_nets_modules: nn.ModuleList  # just to include f_nets to Module parameters

    f_net_norms: List[nn.LayerNorm]

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
        self.f_nets = [nn.Linear(f_net_inp_size, self.hidden_size) for _ in range(self.num_passes)]  # f^i

        """ Normalization """
        self.f_net_norms = [nn.LayerNorm(self.hidden_size) for _ in range(self.num_passes)]
        self.f_nets_modules = nn.ModuleList(self.f_nets + self.f_net_norms)

        self.to(self.device)

    def deserialize(self, data: Dict[str, Any]):
        super().deserialize(data)

        for f_net_id, f_net in enumerate(self.f_nets):
            f_net.load_state_dict(data[f'f_net_{f_net_id}'])
        for f_net_norm_id, f_net_norm in enumerate(self.f_net_norms):
            f_net_norm.load_state_dict(data[f'f_net_norm_{f_net_norm_id}'])

    def serialize(self) -> Dict[str, Any]:
        result = super().serialize()

        for f_net_id, f_net in enumerate(self.f_nets):
            result[f'f_net_{f_net_id}'] = f_net.state_dict()
        for f_net_norm_id, f_net_norm in enumerate(self.f_net_norms):
            result[f'f_net_norm_{f_net_norm_id}'] = f_net_norm.state_dict()

        return result

    def forward(self, observation: Tensor) -> Tensor:
        """Process the input, communicate self.num_passes and compute and return the action"""
        obs_processed = self._process_observation(observation)

        hidden_zero = obs_processed
        hidden = hidden_zero  # initial hidden state is computed from the input
        comm_input = self._zeros()  # initial comm input

        """Do K communication passes"""
        for pass_id, (f_net, f_net_norm) in enumerate(zip(self.f_nets, self.f_net_norms)):

            # build the input to the f^i
            f_inputs = [comm_input, hidden, hidden_zero] if self.skip_connections else [comm_input, hidden]
            f_inputs = torch.cat(f_inputs, dim=1)

            # compute new hidden
            hidden = f_net_norm(self.activation(f_net.forward(f_inputs)))

            # communicate
            if self.comm_enabled and pass_id < (self.num_passes - 1):
                comm_input = self._communicate(hidden, self.comm_mask)

        return self._compute_action(hidden)
