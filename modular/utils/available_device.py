from argparse import Namespace
from typing import Optional

import torch
from badger_utils.sacred import SacredConfigFactory

"""Determine the device automatically by default"""
device_used = 'cuda' if torch.cuda.is_available() else 'cpu'


def choose_device(config: Namespace):
    """Possibility to force_cpu by the experiment config"""
    if config.force_cpu:
        global device_used
        device_used = 'cpu'


def my_device() -> str:
    """Returns the device that should be used in this experiment"""
    return device_used


def get_sacred_storage(storage: Optional[str] = None):
    """Uses contents of the storage_local.txt for determining the storage
     this can be overriden using the method parameter"""
    if storage is not None:
        if storage == 'local':
            return SacredConfigFactory.local()
        elif storage == 'shared':
            return SacredConfigFactory.shared()
        else:
            raise Exception('storage: should be either None, local or shared')

    with open('storage_local.txt', 'r') as file:
        data = file.read()

        if data == 'True':
            return SacredConfigFactory.local()
        elif data == 'False':
            return SacredConfigFactory.shared()

    raise Exception('Could not locate the file storage_local.txt in the project root containing True/False')
