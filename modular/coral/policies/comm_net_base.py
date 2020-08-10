from abc import abstractmethod
from typing import Dict, Any, Optional

import torch
from badger_utils.torch.device_aware import DeviceAwareModule
from torch import Tensor
from torch import nn

from coral.coral_config import CoralConfig
from utils.available_device import my_device
import numpy as np


class CommNetBase(DeviceAwareModule):
    """ CommNetBase, shared by the FF and LSTM versions"""

    c: CoralConfig

    num_experts: int
    batch_size: int
    output_size_per_expert: int
    input_size_per_expert: int
    hidden_size: int

    comm_enabled: bool

    comm_mask: Tensor

    input_net: nn.Linear
    action_net: nn.Linear

    input_net_norm: nn.LayerNorm

    activation: torch.nn

    def __init__(self,
                 batch_size: int,
                 num_experts: int,
                 input_size_per_expert: int,
                 output_size_per_expert: int,
                 conf: CoralConfig):

        super().__init__(device=my_device())

        self.c = conf

        self.num_experts = num_experts
        self.hidden_size = conf.hidden_size

        self.output_size_per_expert = output_size_per_expert
        self.input_size_per_expert = input_size_per_expert
        self.batch_size = batch_size

        self.comm_enabled = conf.comm_enabled
        self.num_passes = conf.num_passes if conf.num_passes > 0 else num_experts

        """ Networks """
        self.input_net = nn.Linear(input_size_per_expert, self.hidden_size)  # encode input to h^0
        self.action_net = nn.Linear(self.hidden_size, self.output_size_per_expert)  # produces actions at the end

        """ Normalization """
        self.input_net_norm = nn.LayerNorm(self.hidden_size)

        self.activation = torch.nn.Tanh()
        self.comm_mask = self.make_fc_mask()

    def _zeros(self, batch_size: Optional[int] = None) -> Tensor:
        batch_size = batch_size if batch_size is not None else self.batch_size
        return torch.zeros(batch_size * self.num_experts, self.hidden_size, device=self.device)

    def _rand(self, batch_size: Optional[int] = None) -> Tensor:
        batch_size = batch_size if batch_size is not None else self.batch_size
        return torch.rand((batch_size * self.num_experts, self.hidden_size), device=self.device)

    def make_fc_mask(self) -> Tensor:
        """Create the comm. mask of size [batch_size, num_experts, num_experts, hidden_size]"""

        mask = torch.ones(self.num_experts, self.num_experts, device=self.device)
        mask = mask - torch.eye(self.num_experts, device=self.device)
        mask = mask / (self.num_experts - 1)

        mask = mask.view(1, self.num_experts, self.num_experts, 1)
        mask = mask.expand(self.batch_size, -1, -1, self.hidden_size)

        return mask

    def deserialize(self, data: Dict[str, Any]):
        data: Dict[str, Dict[str, torch.Tensor]]

        self.input_net.load_state_dict(data['input_net'])
        self.action_net.load_state_dict(data['action_net'])
        self.input_net_norm.load_state_dict(data['input_net_norm'])

    def serialize(self) -> Dict[str, Any]:
        result = {
            'input_net': self.input_net.state_dict(),
            'action_net': self.action_net.state_dict(),
            'input_net_norm': self.input_net_norm.state_dict(),
        }
        return result

    def _process_observation(self, observation: Tensor) -> Tensor:
        """Process the input to h^0"""
        # partition the input between Experts uniformly [batch_size * num_agents, hidden_size] and process it it
        obs = observation.view(self.num_experts * self.batch_size, self.input_size_per_expert)
        obs_processed = self.input_net_norm(self.activation(self.input_net(obs)))
        return obs_processed

    def _compute_action(self, final_hidden: Tensor) -> Tensor:
        """Compute the action (probabilities)"""
        actions = self.activation(self.action_net.forward(final_hidden))
        return actions.view(self.batch_size, self.num_experts, self.output_size_per_expert)

    @abstractmethod
    def forward(self, observation: Tensor) -> Tensor:
        """Process the input, communicate self.num_passes and compute and return the action"""
        pass

    def _communicate(self, comm_in: Tensor, comm_mask: Tensor) -> Tensor:
        return self.communicate(comm_in, comm_mask, self.batch_size, self.num_experts, self.hidden_size)

    @staticmethod
    def communicate(comm_in: Tensor,
                    comm_mask: Tensor,
                    batch_size: int,
                    num_experts: int,
                    hidden_size: int) -> Tensor:
        """Collect the hidden activations from Experts given by the comm_mask
        Args:
            hidden_size:
            num_experts:
            batch_size:
            comm_mask: communication mask (e.g. "mean from all except me")
            comm_in: input to the communication channel, hidden_size from each Expert
        Returns: result of the communication: hidden_size for each Expert
        """
        comm = comm_in.view(batch_size, 1, num_experts, hidden_size).expand(-1,
                                                                            num_experts,
                                                                            -1,
                                                                            -1)
        comm = comm * comm_mask

        comm = comm.view(batch_size * num_experts, num_experts, hidden_size)
        comm = comm.sum(1)
        return comm

    def reset(self, batch_size: Optional[int] = None):
        pass

    def load_comm_matrix(self, matrix: np.ndarray):
        assert matrix.shape == (self.num_experts, self.num_experts), 'incompatible comm matrix given'

        # line normalize
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        matrix = np.nan_to_num(matrix)  # replace nans with zeros
        mask = torch.from_numpy(matrix).type_as(self.comm_mask)
        mask = mask.view(1, self.num_experts, self.num_experts, 1)
        self.comm_mask = mask.expand(self.batch_size, -1, -1, self.hidden_size).to(self.device)
