import time
from argparse import Namespace
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
from badger_utils.sacred import SacredUtils, SacredReader
from badger_utils.view.observer_utils import CompoundObserver
from sacred import Experiment
from sacred.run import Run

from coral.policies.comm_net_base import CommNetBase
from coral.coral_config import CoralConfig
from coral.environment.coral_env import CoralEnv
from coral.environment.coral_env_wrapper import CoralEnvWrapper
from coral.environment.descriptor_node import Graph
from coral.robot_generators import make_robot, RobotGenerator
from utils.available_device import choose_device, get_sacred_storage, my_device
from utils.utils import locate_class

ex = Experiment("Coral-baseline")
sacred_utils = SacredUtils(get_sacred_storage())
sacred_writer = sacred_utils.get_writer(ex)

SacredUtils.add_config_to_experiment(ex, CoralConfig)


def run_experiment(policy,
                   env,
                   optimizer=None,
                   num_epochs: int = 1,
                   serialization_period: Optional[int] = None,
                   testing_period: int = 1,
                   loss_last_n: Optional[int] = None,
                   reset_policy: bool = True,
                   sleep: Optional[float] = None,
                   c: CoralConfig = None) -> Dict:
    """One method shared by: render, training and test/validate (if needed in the future)"""

    render = sleep is not None
    train = optimizer is not None

    total_steps = 0
    total_episodes = 0
    info = {}

    if isinstance(env, CoralEnvWrapper):
        load_comm_matrix(c.matrix, env.num_segments, env.graph, policy)
    policy.reset()

    for update in range(num_epochs):  # epoch

        # prepare for one episode
        done = False
        total_episodes += 1
        info = {}
        losses = []
        observer = CompoundObserver()

        obs = env.reset()
        if isinstance(env, CoralEnvWrapper):
            load_comm_matrix(c.matrix, env.num_segments, env.graph, policy)
        if reset_policy:
            policy.reset()

        while not done:
            total_steps += 1
            if not train:
                with torch.no_grad():
                    action = policy.forward(obs)
            else:
                action = policy.forward(obs)
            obs, rew, done, info = env.step(action.view(env.action_space.shape))

            if train:
                losses.append(info['loss'])
            # env.render()
            # time.sleep(0.0001) # TODO remove
            if render:
                try:
                    env.render(policy=policy)
                except:
                    env.render()
                time.sleep(sleep)
                # print(f'obs: {obs}')
                # print(f'action: {action}')

        if train:
            if loss_last_n is not None:
                assert loss_last_n <= len(losses)
                losses = losses[-loss_last_n:]
            batch_loss = torch.stack(losses).mean()
            if update % serialization_period == 0 or update == 0:
                sacred_writer.save_model(policy, 'policy', epoch=update)

            print(f'update: {update}; \t\tloss is: {batch_loss.item()}')

            # add metrics from task to observer
            observer.main.import_task_info(info)
            # run validation and testing
            # TODO torchEnv here
            # if update % testing_period == 0:
            #     observer.main.import_task_info(env.run_validation(policy))  #, run_experiment))
            #     observer.main.import_task_info(env.run_testing(policy))  #, run_experiment))
            # log values
            observer.main.add_scalar('loss', batch_loss.item())

            # grads
            if policy.c.log_grads:
                grads = policy.get_grads()
                for key, val in grads.items():
                    observer.main.add_scalar(f'{key}_mean', val[0])
                    observer.main.add_scalar(f'{key}_std', val[1])

            # write observer to sacred
            sacred_writer.save_observer(observer.main, update)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    return info


def above_diag(num_experts: int) -> np.ndarray:
    matrix = np.roll(np.eye(num_experts), -1, axis=0)  # above-diagonal matrix
    matrix[-1, 0] = 0
    return matrix


def below_diag(num_experts: int) -> np.ndarray:
    matrix = np.transpose(above_diag(num_experts))
    return matrix


def get_policy_class(policy_name: str):
    folders = ['coral.policies']
    cls = locate_class(folders, policy_name)
    return cls


def load_comm_matrix(type: str, num_experts: int, graph: Graph, policy: CommNetBase):
    matrix = None
    if type is None:
        pass
    elif type == 'above':
        matrix = above_diag(num_experts)
    elif type == 'below':
        matrix = below_diag(num_experts)
    elif type == 'robot':
        matrix = graph.get_connection_matrix()
    elif type == 'robot_inverse':
        matrix = np.transpose(graph.get_connection_matrix())
    else:
        raise Exception('unexpected type of comm matrix')

    if matrix is not None:
        # print(f'loading custom comm matrix: \n{matrix}')
        policy.load_comm_matrix(matrix)


def tag_experiment(writer, c: CoralConfig):
    """Tag experiment in omniboard"""
    writer.add_tag(f'Coral')
    writer.add_tag(f'{c.policy}')
    writer.add_tag(f'matrix_{c.matrix}')
    writer.set_notes(f'policy:{c.policy}, matrix:{c.matrix}, p_branch:{c.p_branching}')


@ex.automain
def run_exp(_run: Run, _config):
    choose_device(Namespace(**_config))
    c = CoralConfig(**_config)
    tag_experiment(sacred_writer, c)

    if c.type == 'generated':
        generator = RobotGenerator(c.num_segments, c.p_branching, c.min_segment_length, c.angle_range)
        env = CoralEnvWrapper(generator,
                              c=c,
                              ep_len=c.ep_len,
                              randomize=c.randomize,
                              seed=c.seed,
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
                       ep_len=c.ep_len,
                       randomize=c.randomize,
                       seed=c.seed,
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

    # Independent, Shared, Global, (+subpolicy), RnnCommNet, LstmCommNet, LstmCommNetCoordinated, FfCommNet
    policy_cls = get_policy_class(c.policy)
    policy = policy_cls(batch_size, num_experts, obs_size_per_expert, action_size_per_expert, c)

    optimizer = torch.optim.Adam(policy.parameters(), lr=c.lr)

    if c.load_from is not None:
        reader = SacredReader(c.load_from, get_sacred_storage(), data_dir=Path.cwd())
        epoch = reader.find_last_epoch()
        print(f'Deserialization from the epoch: {epoch}')
        reader.load_model(policy, 'policy', epoch=epoch)

    # train
    run_experiment(policy,
                   env,
                   sleep=c.sleep,
                   optimizer=optimizer,
                   num_epochs=c.epochs,
                   reset_policy=c.reset_policy,
                   loss_last_n=c.loss_last_n,
                   serialization_period=c.serialization_period,
                   testing_period=c.testing_period,
                   c=c)

    print(f'Training done')
