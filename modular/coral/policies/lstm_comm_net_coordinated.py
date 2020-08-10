from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn

from coral.coral_config import CoralConfig
from coral.policies.lstm_comm_net import LstmCommNet


class LstmCommNetCoordinated(LstmCommNet):
    """ Does the following:
        - first coord_steps * num_experts * (num_experts - 1) tries to estimate the comm. matrix:
            - each source expert is ran with every target expert coord_steps times
            - during this phase, the expert produces random actions
            - a coord_net receives actions and observation of the target expert
            - a the end of the phase produces binary output -> value to the comm_matrix
        - then a normal LstmCommNet run continues
        """

    coord_steps: int
    step: int

    comm_matrix: Tensor

    coord_net: nn.LSTM
    coord_out: nn.Linear
    coord_act: nn

    coord_hidden: Tuple[Tensor, Tensor]

    prev_action: Tensor
    prev_observation: Tensor  # prev observation during the coord phase
    expanded: bool

    coord_layers: int = 1

    sum_diff: Tensor
    use_net: bool
    binary_threshold: bool

    def __init__(self,
                 batch_size: int,
                 num_experts: int,
                 input_size_per_expert: int,
                 output_size_per_expert: int,
                 conf: CoralConfig):

        super().__init__(batch_size, num_experts, input_size_per_expert, output_size_per_expert, conf)

        self.coord_steps = conf.coord_steps

        self.coord_net = nn.LSTM(input_size=output_size_per_expert + input_size_per_expert,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.coord_layers)
        self.coord_out = nn.Linear(self.hidden_size, 1)
        self.coord_act = torch.nn.Sigmoid()
        self.use_net = conf.use_net
        self.binary_threshold = conf.binary_threshold

        self.reset()
        self.to(self.device)

    def deserialize(self, data: Dict[str, Any]):
        super().deserialize(data)

    def serialize(self) -> Dict[str, Any]:
        result = super().serialize()
        return result

    @property
    def is_coord_phase(self) -> bool:
        return self.step < self.coord_steps * self.num_experts * self.num_targets

    @property
    def num_targets(self) -> int:
        return self.num_experts - 1

    @property
    def expert_indexes(self) -> Tuple[int, int, bool, bool]:
        """Get current source_id and target_id in the coord phase"""
        run_phase = self.step // self.coord_steps

        source_id = run_phase // self.num_targets
        target_id = run_phase % self.num_targets
        should_reset = self.step % self.coord_steps == 0
        should_collect = ((self.step + 1) // self.coord_steps) != run_phase

        targets = list(range(self.num_experts))
        targets.pop(source_id)  # exclude the source id
        return source_id, targets[target_id], should_reset, should_collect

    def coord_forward(self, data: Tensor) -> Tensor:
        data = data.unsqueeze(0)
        output, self.coord_hidden = self.coord_net.forward(data, self.coord_hidden)
        out = self.coord_act(self.coord_out.forward(output))
        return self.coord_act(out)

    def forward(self, observation: Tensor) -> Tensor:
        self.step += 1

        # observation: [batch_size, num_expert, input_size_per_expert]
        if self.is_coord_phase:
            source_id, target_id, should_reset, should_collect = self.expert_indexes

            if should_reset:
                # TODO store the last result
                # print(f'---------- reset')
                self._clear_hidden()
                self.sum_diff.zero_()
                self.prev_observation = observation[:, target_id]  # should start with the current one, so that diff is 0

            # coord_in = torch.cat([observation[:, target_id], self.prev_action[:, source_id]], dim=1)
            obs = self.prev_observation - observation[:, target_id]

            if self.use_net:
                coord_in = torch.cat([obs, torch.zeros_like(self.prev_action[:, source_id])], dim=1)  # actions are ingnored now
                act = self.coord_forward(coord_in)
            else:
                abs_obs = torch.abs(obs).mean(dim=1).view(-1, 1)
                self.sum_diff = self.sum_diff + abs_obs

            self.prev_observation = observation[:, target_id]

            # print(f'coord, step {self.step},\t src:target {source_id}:{target_id}')
            action = torch.zeros((self.batch_size, self.num_experts, self.output_size_per_expert), device=self.device)
            action[:, source_id] = (torch.rand(self.batch_size, self.output_size_per_expert, device=self.device) - 0.5) * 2
            self.prev_action = action

            if should_collect:
                # print(f'collecting here!!!!!!!!')
                if self.use_net:
                    self.comm_matrix[:, source_id, target_id] = act.view(self.batch_size)
                else:
                    self.comm_matrix[:, source_id, target_id] = self.sum_diff.view(self.batch_size)
                # print(f'new comm matrix is: \n {self.comm_matrix.detach().cpu().numpy()}')
            return action
            # return super().forward(observation)
        else:
            if not self.expanded:
                # print(f'expanding comm_matrix to comm_mask \n\n\n\n\n')
                # self.comm_matrix = torch.nn.functional.normalize(self.comm_matrix, p=1, dim=2)
                if self.binary_threshold and not self.use_net:
                    self.comm_matrix = (self.comm_matrix > 0.001).float() * 1
                self.comm_mask = self.comm_matrix.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)

                self.expanded = True
                super().reset()
            # print(f'normal step')
            # self.comm_mask determined, continue normally now
            return super().forward(observation)

    def _clear_hidden(self, batch_size: Optional[int] = None):
        # hidden state of coord LSTM
        batch_size = self.batch_size if batch_size is None else batch_size
        size = (self.coord_layers, batch_size, self.hidden_size)

        self.coord_hidden = (torch.zeros(size, device=self.device),
                             torch.zeros(size, device=self.device))

    def load_comm_matrix(self, matrix: np.ndarray):
        # ignores custom comm. matrix
        pass

    def reset(self, batch_size: Optional[int] = None):
        super().reset(batch_size)

        batch_size = batch_size if batch_size is not None else self.batch_size

        self.step = -1
        # clear the comm_matrix [source, target expert]
        self.comm_matrix = torch.zeros((batch_size, self.num_experts, self.num_experts),
                                       device=self.device)

        self.prev_action = torch.zeros((batch_size, self.num_experts, self.output_size_per_expert),
                                       device=self.device)

        self.prev_observation = torch.zeros((batch_size, self.input_size_per_expert), device=self.device)

        self._clear_hidden(batch_size)
        self.expanded = False
        self.sum_diff = torch.zeros((batch_size, 1), device=self.device)

