import torch
from torch import Tensor

from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.tasks.single_point_reacher import SinglePointReacher


class SinglePointReacherPartial(SinglePointReacher):
    """Reach to a randomly selected point in a 2D space, only one effector sees the goal (others zeros)."""

    hidden_goal: Tensor

    def __init__(self,
                 owner: CoralEnv,
                 c: CoralConfig):
        super().__init__(owner, c)
        self.hidden_goal = torch.zeros((self.batch_size, 2), dtype=torch.float, device=self.device)

    def custom_obs_per_segment(self, segment_id: int) -> Tensor:
        if segment_id == self.num_segments - 1:
            return self.goal_pos
        return self.hidden_goal

