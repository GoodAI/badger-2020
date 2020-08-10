import re
from typing import List, Union, Optional

import torch
from badger_utils.view.bokeh.tensor_plot import TensorPlot
from torch import Tensor
from badger_utils.view.bokeh.bokeh_component import BokehComponent
from badger_utils.view.bokeh_utils import plot_tensor
from badger_utils.view.signals import signal
from bokeh.layouts import row, column
from bokeh.models import Slider, Panel, TextInput, Div


class SliderVectorSignals:
    def __init__(self):
        self.on_changed = signal(Tensor)


class SliderVector(BokehComponent):
    _text: TextInput = None

    def __init__(self, count: int, start: float = -1.0, end: float = 1.0, step: float = 0.1, size: int = 80,
                 decimals: int = 1,
                 values: Union[List[float], Tensor] = None, show_text: bool = False, title: Optional[str] = None):
        self.signals = SliderVectorSignals()
        self._count = count
        self._start = start
        self._end = end
        self._step = step
        self._size = size
        self._decimals = decimals
        self._title = title
        if isinstance(values, Tensor):
            values = values.view(-1).tolist()
        self._values = values or [0.0] * count
        assert len(self._values) == count
        self.sliders = [self._create_slider(i) for i in range(count)]
        self._plot_tensor = TensorPlot(torch.tensor(self._values))
        if show_text:
            self._text = TextInput(width=27*count)
            self._text.on_change('value', lambda a, o, n: self._update_by_text(n))
        self.update_vector()

    def _create_slider(self, i: int) -> Slider:
        str_format = f'0.{"".join(["0"] * self._decimals)}'
        slider = Slider(start=self._start, end=self._end, step=self._step, value=self._values[i],
                        orientation='vertical', format=str_format, direction='rtl',
                        default_size=self._size)
        slider.on_change('value', lambda a, o, n: self.update_vector())
        return slider

    @property
    def value(self) -> Tensor:
        values = [float(s.value) for s in self.sliders]
        return torch.tensor(values)

    def update_vector(self):
        tensor = self.value
        self._plot_tensor.update(tensor)
        self.signals.on_changed.emit(tensor)

    def create_layout(self):
        items = [row(*self.sliders), self._plot_tensor.create_layout()]
        if self._text is not None:
            items = [self._text] + items
        if self._title is not None:
            items = [Div(text=self._title)] + items
        return column(*items)

    def _update_by_text(self, value: str):
        values = [float(v.strip()) for v in re.split(r'[,\s]+', value)]
        for i, val in enumerate(values):
            # for slider, val in zip(self.sliders, values):
            self.sliders[i].value = val

    def set_text(self, text: str):
        self._text.value = text