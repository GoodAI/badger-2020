from pathlib import Path

import torch

from badger_utils.torch.serializable import Serializable
from badger_utils.view.observer_utils import Observer
from git import GitConfigParser
from matplotlib.figure import Figure
from sacred.host_info import host_info_gatherer
from torch import Tensor
from PIL import Image
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from badger_utils.file.temp_file import TempFile, TempFileName
from badger_utils.sacred.sacred_config import SacredConfig
from badger_utils.sacred.sacred_rw_base import SacredRWBase
import os


class SacredWriter(SacredRWBase):
    """Writes data to the Sacred"""

    _experiment: Experiment

    def __init__(self, experiment: Experiment, sacred_config: SacredConfig):
        super().__init__()
        self._mongo_observer = sacred_config.create_mongo_observer()
        self._experiment = experiment
        self._experiment_setup()

    def set_notes(self, notes: str):
        """ Set omniboard notes to experiment """
        run_id = self.experiment_id
        self._mongo_observer.runs.update_one({'_id': run_id}, {'$set': {'omniboard.notes': notes}})

    def add_tag(self, tag: str):
        """ Add omniboard tag to experiment """
        run_id = self.experiment_id
        self._mongo_observer.runs.update_one({'_id': run_id}, {'$push': {'omniboard.tags': tag}})

    @property
    def experiment_id(self) -> int:
        # noinspection PyProtectedMember
        try:
            return self._experiment.current_run._id
        except AttributeError:
            raise ValueError(
                'Experiment id is not set yet. Possible cause - usage of experiment id outside of main function.')

    def save_tensor(self, data: Tensor, name: str, epoch: int):
        with TempFile(self.create_artifact_name(f'{name}.pt', epoch)) as file:
            with file.open('wb') as f:
                torch.save(data, f)
            self._add_binary_file(file)

    def save_model(self, model: Serializable, name: str, epoch: int):
        dictionary = model.serialize()
        blob = self._dict_to_blob(dictionary)

        # create temp directory, write blob to the file, push file to the sacred, delete directory
        with TempFile(self.create_artifact_name(name + SacredRWBase.MODEL_SUFFIX, epoch)) as file:
            with file.open('wb') as f:
                f.write(blob)
            self._add_binary_file(file)

    def save_scalar(self, value: float, name: str, epoch: int):
        self._experiment.log_scalar(name, value, epoch)

    # noinspection PyShadowingBuiltins
    def save_matplot_figure(self, figure: Figure, name: str, epoch: int, format: str = 'svg'):
        with TempFileName(self.create_artifact_name(name, epoch)) as filename:
            figure.savefig(fname=filename, format=format)
            self._experiment.add_artifact(filename)

    def save_image(self, image: Image, name: str, epoch: int):
        with TempFile(self.create_artifact_name(name, epoch)) as file:
            with file.open('wb') as f:
                image.save(f)
            self._experiment.add_artifact(str(file))

    def _experiment_setup(self):
        self._experiment.captured_out_filter = lambda captured_output: "Output capturing turned off."
        self._experiment.observers.append(self._mongo_observer)
        for i in [SacredWriter._git_user, SacredWriter._git_email]:
            self._experiment.additional_host_info.append(i)

    def _add_binary_file(self, file: Path):
        self._experiment.add_artifact(filename=str(file), content_type='application/octet-stream')

    def save_observer(self, observer: Observer, epoch: int):
        for s in observer.scalars:
            self.save_scalar(s.value, s.name, epoch)
        for t in observer.tensors:
            self.save_tensor(t.value, t.name, epoch)
        for p in observer.plots:
            self.save_matplot_figure(p.figure, p.name, epoch)
        for p in observer.models:
            self.save_model(p.value, p.name, epoch)
        self._experiment.current_run.info['environment_data'] = observer.environment_data

    @staticmethod
    def _get_config_path():
        return os.path.join(os.environ.get("HOME", '~'), ".gitconfig")

    @staticmethod
    @host_info_gatherer(name="git_user")
    def _git_user():
        config = GitConfigParser(SacredWriter._get_config_path())
        return config.get_value('user', 'name', '---')

    @staticmethod
    @host_info_gatherer(name="git_email")
    def _git_email():
        config = GitConfigParser(SacredWriter._get_config_path())
        return config.get_value('user', 'email', '---')
