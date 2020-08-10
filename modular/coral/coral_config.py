from dataclasses import dataclass
from typing import Optional


@dataclass
class CoralConfig:

    # common
    seed: int  # TODO not supported everywhere
    force_cpu: bool = False  # if GPU found, it is used automatically, override if necessary
    sleep: Optional[float] = None  # set this to float in order to redner during training (debug)
    serialization_period: int = 500
    testing_period: int = 100

    # policy
    policy: str = 'FfCommNet'  # FfCommNet, LstmCommNet, RnnCommNet, LstmCommNetCoordinated, Global, Shared, Independent
    subpolicy: str = 'LSTMNet'
    hidden_size: int = 128
    num_layers: int = 2
    rand_hidden: bool = False
    reset_policy: bool = True  # reset policy on env.done? TODO does not work yet

    log_grads: bool = False  # not tested

    # learning
    epochs: int = 4000
    lr: float = 0.0001
    batch_size: int = 128

    # env
    ep_len: int = 70
    loss_last_n: Optional[int] = None  # loss computed from the last n steps of the episode/rollout
    randomize: bool = False  # random segment length
    fixed_goal: bool = False  # only one position of goal
    task: str = 'MultiPointReacher'  # name of the task class to be ran
    rotation_limit: Optional[float] = None  # angle = angle + action * cos(angle * rotation_limit)
    acf: str = 'gelu'  # activation function for the runner
    init_noise: float = 0.0  # noise applied to the initial conditions

    # manipulator task
    max_goal_dist: float = 0.5  # 1 is the length of the longest part of the robot
    act_force: str = 'normal'  # TODO remove
    num_points: int = 2

    # guide_runner
    target_duration: int = 30
    goals_hidden: bool = False

    # fractal generator params
    turn: float = 33  # turn angle of the fractal
    iterations: int = 2  # no. iterations during the fractal generation
    type: str = 'generated'  # type of the fractal, types a, b, linear, fork

    # robot generator params
    # if type==generated, these are used
    num_segments: int = 8  # number of robot segments (if generated)
    p_branching: float = 0  # probability of branching of the robot
    min_segment_length: float = 1  # minimum segment length (max is 1 before the robot normalization)
    angle_range: float = 0.1  # from which range to sample the static rotation of the joint? (radians)

    load_from: Optional[int] = None

    # commnet
    comm_enabled: bool = True
    num_passes: int = -1  # if -1, the num_passes should be automatically derived from num_experts
    skip_connections: bool = True  # stack observation next to the hidden state each comm pass?
    matrix: Optional[str] = None  # load the comm. matrix from?

    # LstmCommNetCoordinated
    coord_steps: int = 3  # num coordination steps per robot segment pair
    use_net: bool = False
    binary_threshold: bool = True  # threshold the values of the comm_matrix after the coord phase to 0/1?

