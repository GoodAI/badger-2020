from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Dict, Set, Optional, Any

from badger_utils.torch.serializable import Serializable
from matplotlib.figure import Figure
from torch import Tensor
import pandas as pd


@dataclass
class ObserverPlot:
    name: str
    figure: Figure


@dataclass
class ObserverScalar:
    name: str
    value: float


@dataclass
class ObserverTensor:
    name: str
    value: Tensor


@dataclass
class ObserverModel:
    name: str
    value: Serializable


class ObserverLevel(IntEnum):
    training = 1
    testing = 2
    inference = 3

    @classmethod
    def should_log(cls, run_level: Optional['ObserverLevel'], log_level: Optional['ObserverLevel']):
        return run_level is None or log_level is None or run_level >= log_level


class Observer(ABC):
    @property
    @abstractmethod
    def plots(self) -> List[ObserverPlot]:
        pass

    @property
    @abstractmethod
    def scalars(self) -> List[ObserverScalar]:
        pass

    @property
    @abstractmethod
    def tensors(self) -> List[ObserverTensor]:
        pass

    @property
    @abstractmethod
    def models(self) -> List[ObserverModel]:
        pass

    @property
    @abstractmethod
    def environment_data(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def add_plot(self, name: str, figure: Figure, log_level: Optional[ObserverLevel] = None):
        pass

    @abstractmethod
    def add_scalar(self, name: str, value: float, log_level: Optional[ObserverLevel] = None):
        pass

    @abstractmethod
    def add_tensor(self, name: str, tensor: Tensor, log_level: Optional[ObserverLevel] = None):
        pass

    @abstractmethod
    def add_model(self, name: str, model: Serializable, log_level: Optional[ObserverLevel] = None):
        pass

    def tensors_as_dict(self) -> Dict[str, Tensor]:
        return {t.name: t.value for t in self.tensors}

    def scalars_as_dict(self) -> Dict[str, float]:
        return {t.name: t.value for t in self.scalars}

    def with_suffix(self, suffix: str) -> 'Observer':
        return SuffixObserver(self, suffix)

    def import_task_info(self, info: Dict):
        for name, value in info.get('metrics', {}).items():
            self.add_scalar(name, value)
        if 'environment_data' in info:
            self.environment_data.update(info['environment_data'])


class ObserverImpl(Observer):
    _plots: List[ObserverPlot]
    _scalars: List[ObserverScalar]
    _tensors: List[ObserverTensor]
    _models: List[ObserverModel]
    _environment_data: Dict[str, Any]

    def __init__(self, run_level: Optional[ObserverLevel] = None, run_flags: Set = None):
        self.run_level = run_level
        self.run_flags = run_flags or set()
        self._plots = []
        self._scalars = []
        self._tensors = []
        self._models = []
        self._environment_data = {}

    @property
    def plots(self) -> List[ObserverPlot]:
        return self._plots

    @property
    def scalars(self) -> List[ObserverScalar]:
        return self._scalars

    @property
    def tensors(self) -> List[ObserverTensor]:
        return self._tensors

    @property
    def models(self) -> List[ObserverModel]:
        return self._models

    @property
    def environment_data(self) -> Dict[str, Any]:
        return self._environment_data

    def add_plot(self, name: str, figure: Figure, log_level: Optional[ObserverLevel] = None):
        if ObserverLevel.should_log(self.run_level, log_level):
            self.plots.append(ObserverPlot(name, figure))

    def add_scalar(self, name: str, value: float, log_level: Optional[ObserverLevel] = None):
        if ObserverLevel.should_log(self.run_level, log_level):
            self.scalars.append(ObserverScalar(name, value))

    def add_tensor(self, name: str, tensor: Tensor, log_level: Optional[ObserverLevel] = None):
        if ObserverLevel.should_log(self.run_level, log_level):
            self.tensors.append(ObserverTensor(name, tensor))

    def add_model(self, name: str, model: Serializable, log_level: Optional[ObserverLevel] = None):
        if ObserverLevel.should_log(self.run_level, log_level):
            self.models.append(ObserverModel(name, model))

    def tensors_as_dict(self) -> Dict[str, Tensor]:
        return {t.name: t.value for t in self.tensors}

    def scalars_as_dict(self) -> Dict[str, float]:
        return {t.name: t.value for t in self.scalars}

    def with_suffix(self, suffix: str) -> 'Observer':
        return SuffixObserver(self, suffix)

    def import_task_info(self, info: Dict):
        for name, value in info.get('metrics', {}).items():
            self.add_scalar(name, value)
        if 'environment_data' in info:
            self.environment_data.update(info['environment_data'])


class ObserverWrapper(Observer):
    _observer: Observer

    def __init__(self, observer: Observer):
        super().__init__()
        self._observer = observer

    @property
    def plots(self) -> List[ObserverPlot]:
        return self._observer.plots

    @property
    def scalars(self) -> List[ObserverScalar]:
        return self._observer.scalars

    @property
    def tensors(self) -> List[ObserverTensor]:
        return self._observer.tensors

    @property
    def models(self) -> List[ObserverModel]:
        return self._observer.models

    @property
    def environment_data(self) -> Dict[str, Any]:
        return self._observer.environment_data

    def add_plot(self, name: str, figure: Figure, log_level: Optional[ObserverLevel] = None):
        self._observer.add_plot(self._process_name(name), figure, log_level)

    def add_scalar(self, name: str, value: float, log_level: Optional[ObserverLevel] = None):
        self._observer.add_scalar(self._process_name(name), value, log_level)

    def add_tensor(self, name: str, tensor: Tensor, log_level: Optional[ObserverLevel] = None):
        self._observer.add_tensor(self._process_name(name), tensor, log_level)

    def add_model(self, name: str, model: Serializable, log_level: Optional[ObserverLevel] = None):
        self._observer.add_model(self._process_name(name), model, log_level)

    def _process_name(self, name: str) -> str:
        return name


@dataclass
class FilterObserverConfig:
    log_scalar: bool = True
    log_tensor: bool = True
    log_model: bool = True
    log_plot: bool = True

    @staticmethod
    def from_periods(epoch: int, scalar: int = 1, tensor: int = 1, model: int = 1,
                     plot: int = 1) -> 'FilterObserverConfig':
        return FilterObserverConfig(
            log_scalar=epoch % scalar == 0,
            log_tensor=epoch % tensor == 0,
            log_model=epoch % model == 0,
            log_plot=epoch % plot == 0
        )


_filter_observer_config_all = FilterObserverConfig()


class FilterObserver(ObserverWrapper):
    def __init__(self, observer: Observer, config: FilterObserverConfig):
        super().__init__(observer)
        self._config = config

    def add_plot(self, name: str, figure: Figure, log_level: Optional[ObserverLevel] = None):
        if self._config.log_plot:
            super().add_plot(name, figure, log_level)

    def add_scalar(self, name: str, value: float, log_level: Optional[ObserverLevel] = None):
        if self._config.log_scalar:
            super().add_scalar(name, value, log_level)

    def add_tensor(self, name: str, tensor: Tensor, log_level: Optional[ObserverLevel] = None):
        if self._config.log_tensor:
            super().add_tensor(name, tensor, log_level)

    def add_model(self, name: str, model: Serializable, log_level: Optional[ObserverLevel] = None):
        if self._config.log_model:
            super().add_model(name, model, log_level)


class SuffixObserver(ObserverWrapper):
    def __init__(self, observer: Observer, suffix: str):
        super().__init__(observer)
        self._suffix = suffix

    def _process_name(self, name: str) -> str:
        return name + self._suffix


class MultiObserver:
    observers: List[Observer]

    def __init__(self, run_level: Optional[ObserverLevel] = None, add_observer: bool = True):
        self._run_level = run_level
        self.observers = []
        if add_observer:
            self.add_observer()

    @property
    def current(self) -> Observer:
        return self.observers[-1]

    def add_observer(self) -> Observer:
        observer = ObserverImpl(self._run_level)
        self.observers.append(observer)
        return observer

    def scalars_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([o.scalars_as_dict() for o in self.observers])


class CompoundObserver:
    main: Observer
    rollout: MultiObserver

    def __init__(self, run_level: Optional[ObserverLevel] = None,
                 config: FilterObserverConfig = _filter_observer_config_all):
        self.main = FilterObserver(ObserverImpl(run_level), config)

        # TODO solve rollout observers
        self.rollout = MultiObserver(run_level, add_observer=False)
