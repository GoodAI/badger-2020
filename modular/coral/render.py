from argparse import Namespace
from pathlib import Path
from typing import Optional

import click
from badger_utils.sacred import SacredReader

from coral.baseline_experiment import get_policy_class, run_experiment
from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.coral_env_wrapper import CoralEnvWrapper
from coral.robot_generators import make_robot, RobotGenerator
from utils.available_device import get_sacred_storage, choose_device, my_device


@click.command()
@click.argument('exp_id', type=int)
@click.option('--ep', default=None, help='Episode to be deserialized (the last one by default)')
@click.option('--sleep', default=0.01, help='Sleep time between the time steps')
@click.option('--ep_len', default=None, type=int, help='Override the ep_len?')
@click.option('--seed', default=None, type=int, help='Override the c.seed?')
@click.option('--iters', default=None, type=int, help='Override the c.n_joints?')
@click.option('--num_segments', default=None, type=int, help='Override the c.num_segments for generated robots?')
@click.option('--type', default=None, type=str, help='Override the c.n_joints?')
@click.option('--num_passes', default=None, type=int, help='Override num_passes in the comm_net')
@click.option('--p_branching', default=None, type=float, help='Override p_branching in the comm_net')
@click.option('--storage', default=None, type=str,
              help='Choose the storage different than in storage_local.txt, either local/shared')
def render(exp_id: int,
           ep: int,
           sleep: float,
           ep_len: Optional[int],
           seed: Optional[int],
           iters: Optional[int],
           type: Optional[str],
           num_passes: Optional[int],
           num_segments: Optional[int],
           p_branching: Optional[float],
           storage: Optional[str]):
    """Download a given config and policy from the sacred, run the inference"""

    reader = SacredReader(exp_id, get_sacred_storage(storage), data_dir=Path.cwd())
    choose_device(Namespace(**reader.config))

    c = CoralConfig(**reader.config)
    # tag_experiment(sacred_writer, c)

    ep_len = ep_len if ep_len is not None else c.ep_len
    seed = seed if seed is not None else c.seed
    iters = iters if iters is not None else c.iterations
    type = type if type is not None else c.type
    c.num_passes = num_passes if num_passes is not None else c.num_passes
    c.num_segments = num_segments if num_segments is not None else c.num_segments
    c.p_branching = p_branching if p_branching is not None else c.p_branching

    if c.type == 'generated':
        generator = RobotGenerator(c.num_segments, c.p_branching, c.min_segment_length, c.angle_range)
        env = CoralEnvWrapper(generator,
                              c=c,
                              ep_len=ep_len,
                              randomize=c.randomize,
                              seed=seed,
                              fixed_goal=c.fixed_goal,
                              batch_size=c.batch_size,
                              task_name=c.task,
                              device=my_device(),
                              init_noise=c.init_noise,
                              rotation_limit=c.rotation_limit)
    else:
        graph = make_robot(c)

        env = CoralEnv(graph,
                       c=c,
                       ep_len=ep_len,
                       randomize=c.randomize,
                       seed=seed,
                       fixed_goal=c.fixed_goal,
                       batch_size=c.batch_size,
                       task_name=c.task,
                       device=my_device(),
                       init_noise=c.init_noise,
                       rotation_limit=c.rotation_limit)

    # read dims
    batch_size, num_experts, action_size_per_expert = env.action_space.shape
    _, _, obs_size_per_expert = env.observation_space.shape
    assert action_size_per_expert == 1

    policy_cls = get_policy_class(c.policy)
    policy = policy_cls(batch_size, num_experts, obs_size_per_expert, action_size_per_expert, c)

    # load_comm_matrix(c.matrix, num_experts, graph, policy) # TODO

    epoch = ep if ep is not None else reader.find_last_epoch()
    print(f'Deserialization from the epoch: {epoch}')
    reader.load_model(policy, 'policy', epoch=epoch)

    # train
    run_experiment(policy,
                   env,
                   reset_policy=c.reset_policy,
                   num_epochs=10000,
                   sleep=sleep,
                   c=c)


if __name__ == '__main__':
    render()
