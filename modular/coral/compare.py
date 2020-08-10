import os
from typing import List, Optional, Dict, Tuple
from typing import Union

import click
import matplotlib.pyplot as plt
from badger_utils.sacred import SacredUtils

from utils.available_device import get_sacred_storage
from utils.plot_utils import read_metric
import numpy as np

FOLDER = 'coral/img'


def smooth(data: np.ndarray, window_size: int):
    """1D convolution over the data,
     compared to np.convolve, this adaptively increases the size of the window at the beginning,
     to preserve details where needed"""
    result = np.zeros(data.size - window_size)
    window = np.ones((window_size,))

    for pos in range(result.size):
        win_size = min(pos + 1, window_size)

        if win_size < window_size:
            win = np.ones((win_size,))
        else:
            win = window

        result[pos] = np.sum(data[pos + 1 - win_size:pos + 1] * win) / win.size
    return result


def mean(data: List[float]) -> float:
    return sum(data) / len(data)

def aggregate_vals(all_lines: List[np.ndarray], func) -> np.ndarray:
    """An inefficient way to aggregate given data.

    The all_lines: is a list of series of potentially different length
    return aggregated results from these lines, aggregation done by func (min, max, mean..)
    """
    max_len = max([len(line) for line in all_lines])
    results = []

    for pos in range(max_len):
        temp = []
        for line in all_lines:
            if pos < line.size:
                temp.append(line[pos])

        res = func(temp)
        results.append(res)

    return np.array(results)


def plot_runs(run_ids: List[int],
              label: str,
              metrics: List[str] = ['batch_loss'],
              window: int = 20,
              include_min_max: bool = True,
              include_all: bool = False,
              storage: Optional[str] = None):
    all_lines = []

    for run_id in run_ids:
        print(f'Reading metric from {run_id}..')
        data = read_metric(run_id, f'{metrics[0]}', storage=storage).to_numpy()
        assert data.shape[1] == 1
        all_lines.append(data.reshape(-1))

    print(f'Data loaded')
    all_lines = [smooth(line, window) for line in all_lines]

    min_vals = aggregate_vals(all_lines, min)
    max_vals = aggregate_vals(all_lines, max)
    mean_vals = aggregate_vals(all_lines, mean)

    gens = np.arange(mean_vals.size)

    mean_line = plt.plot(mean_vals, alpha=0.01)[0]
    if include_min_max:
        plt.fill_between(x=gens, y1=min_vals, y2=max_vals, alpha=0.25)
    if include_all:
        for line in all_lines:  # plot all lines, not the resized version
            plt.plot(line, color=mean_line.get_color(), alpha=0.6, linewidth=0.5)
    mean_line = plt.plot(mean_vals, color=mean_line.get_color(), alpha=0.8)[0]
    mean_line.set_label(label)

    print(f'done, loaded {len(run_ids)} of series under label: {label}')


def _should_add(config: Dict, config_vals: Dict) -> bool:
    for key, val in config_vals.items():
        if config[key] != val:
            return False
    return True


def filter_ids_for(all_ids: List[int], config_vals: Dict, sacred_utils: SacredUtils) -> List[int]:
    result = []
    for run_id in all_ids:
        reader = sacred_utils.get_reader(run_id)
        try:
            conf = reader.config  # config might not exist
            if _should_add(reader.config, config_vals):
                result.append(run_id)
        except:
            continue

    return result


def remove_nonexistent(ids: List[int], sacred_utils: SacredUtils) -> List[int]:
    """Remove nonexistent exp ids"""
    result = []
    for run_id in ids:
        reader = sacred_utils.get_reader(run_id)
        try:
            conf = reader.config  # config might not exist
            result.append(run_id)
        except:
            continue
    return result


def get_conf(expid: int, sacred_utils: SacredUtils) -> Dict:
    return sacred_utils.get_reader(expid).config


def get_diffs(uniques: List[Tuple[Dict, List[int]]]) -> List[Dict]:
    """Get the unique configs, return dictionaries with non-constant values"""
    first = uniques[0][0]

    key_diff = []

    for config, _ in uniques[1:]:
        for k, v in first.items():
            if k not in config:
                key_diff.append(k)
            elif first[k] != config[k]:
                key_diff.append(k)

    key_diff = list(set(key_diff))

    print(f'non-constant keys are: {key_diff}')
    result = []
    for unique, _ in uniques:
        result.append({})
        for key in key_diff:
            result[-1][key] = unique[key]

    print(f'diffed dictionaries are: {result}')
    return result


def get_unique_configs(ids: List[int], sacred_utils: SacredUtils) -> List[Tuple[Dict, List[int]]]:
    """Given list of exp IDs, return list of tuples: (config, ids_with_the_config)

     Note: seeds are removed from this!
    """
    uniques = [(get_conf(ids[0], sacred_utils), [ids[0]])]
    uniques[0][0]['seed'] = None

    for expid in ids[1:]:
        config = get_conf(expid, sacred_utils)
        config['seed'] = None
        found = False
        for conf, conf_ids in uniques:
            # code ignores keys that are not everywhere (but splits according to them!)
            if config == conf:
                # if len(set(conf.items()) ^ set(config.items())) == 0:
                conf_ids.append(expid)
                found = True
                continue
        if not found:
            uniques.append((config, [expid]))

    return uniques


def merge(filtera: Dict, filter: Optional[Dict] = None):
    if filter is None:
        return filtera
    return {**filter, **filtera}


def auto_compare(start: int,
                 end: int,
                 name: str,
                 scale: str,
                 filter: Dict,
                 show_ids: bool,
                 showall: bool,
                 metric=['mse'],
                 win=80,
                 folder: str = FOLDER,
                 include_min_max: bool = True,
                 storage: Optional[str] = None):
    """ Automatic comparison of measured metric:

        -Get the EXP_IDs in a given [start,end] range,
        -filter for required config key/values
        -split based on the config diff (group lines with the same config)
        -for each group: print mean, min/max and show the legend with config
    """
    all_ids = list(range(start, end + 1))

    sacred_utils = SacredUtils(get_sacred_storage(storage))

    filtered = filter_ids_for(all_ids, filter, sacred_utils)  # filter by the config values
    print(f'After filtering, EXP_IDs are: {filtered}')

    unique_configs = get_unique_configs(filtered, sacred_utils)
    diffs = get_diffs(unique_configs)

    print(f'Unique configs are: {unique_configs}')
    print(f'num configs is: {len(unique_configs)}')
    print(f'diffs are {diffs}')

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(title=name,
                         xlabel='Learning step',
                         ylabel=f'metric: {metric}',
                         yscale=scale)

    for (config, ids), diffs in zip(unique_configs, diffs):
        legend = str(diffs)
        if show_ids:
            legend += f'  IDs: {str(ids)}'

        plot_runs(ids,
                  label=legend,
                  metrics=metric,
                  include_all=showall,
                  window=win,
                  include_min_max=include_min_max,
                  storage=storage)

    plt.legend()
    plt.savefig(f'{folder}/{name}_from_{start}_to_{end}.png', format='png')
    plt.show()


def get_value(value: str) -> Union[str, float, int]:
    try:
        v = int(value)
        return v
    except ValueError:
        try:
            v = float(value)
            return v
        except ValueError:
            if value == 'True':
                return True
            elif value == 'False':
                return False
            return value


@click.command()
@click.argument('start')
@click.argument('end')
@click.option('--scale', default='linear', help='either linear or log')
@click.option('--ids', default='True', help='show EXP_IDs in legend?')
@click.option('--showall', default='True', help='show all lines?')
@click.option('--showminmax', default='True', help='show min/max range?')
@click.option('--metric', default='loss', help='metric name to show')
@click.option('--win', default=80, help='window len')
@click.option('--title', default='Runs compared', help='graph title')
@click.option('--filter', default=None, help='filter runs with only this key:val config param')
@click.option('--storage', default=None, type=str,
              help='Choose the storage different than in storage_local.txt, either local/shared')
def compare(start: int,
            end: int,
            metric: str,
            scale: str,
            ids: str,
            showall: str,
            win: int,
            title: str,
            showminmax: bool,
            filter: Optional[str],
            storage: Optional[str]):
    """Automatically compare runs from a given range of EXP_IDs and show the measured metrics"""

    if filter is None:
        parsed_filter = {}
    else:
        parsed_filter = {}
        list_keyval = filter.split(',')
        for keyval in list_keyval:
            splitted = keyval.split(':')
            assert len(splitted) == 2, 'expected format of the filter: key:value,key2:value2...'
            value = get_value(splitted[1])
            parsed_filter[splitted[0]] = value
        print(f'will filter IDs with this filter: {parsed_filter}')

    ids = ids == 'True'
    showall = showall == 'True'
    start = int(start)
    end = int(end)
    auto_compare(start,
                 end,
                 name=title,
                 scale=scale,
                 filter=parsed_filter,
                 show_ids=ids,
                 showall=showall,
                 win=win,
                 metric=[metric],
                 folder=os.getcwd() + '/img/',
                 include_min_max=showminmax == 'True',
                 storage=storage)


if __name__ == '__main__':
    """Compare various runs and show in one graph"""
    compare()
