import re
from typing import List, Union, Optional

import torch
from badger_utils.view.bokeh.tensor_plot import TensorPlot
from torch import Tensor
from badger_utils.view.bokeh.bokeh_component import BokehComponent
from badger_utils.view.bokeh_utils import plot_tensor
from badger_utils.view.signals import signal
from bokeh.layouts import row, column
from bokeh.models import Slider, Panel, TextInput, Div, Tabs, LayoutDOM


class SliderTensorSignals:
    def __init__(self):
        self.on_changed = signal(Tensor)


class SliderTensor(BokehComponent):
    _text: TextInput = None
    _internal_update = False

    def __init__(self, shape: List[int], start: float = -1.0, end: float = 1.0, step: float = 0.1, size: int = 80,
                 decimals: int = 1,
                 initial_value: Tensor = None, show_text: bool = False, title: Optional[str] = None,
                 slider_orientation: str = 'horizontal'):
        self._slider_orientation = slider_orientation
        self.signals = SliderTensorSignals()
        assert len(shape) == 2, 'Only 2D Tensors are supported for now'
        self._shape = shape
        self._start = start
        self._end = end
        self._step = step
        self._size = size
        self._decimals = decimals
        self._title = title
        self._tensor = initial_value if initial_value is not None else torch.zeros(shape)
        count = self._tensor.numel()
        self.sliders = [self._create_slider(i) for i in range(count)]
        self.texts = [self._create_text(i) for i in range(count)]
        self._plot_tensor = TensorPlot(torch.tensor(self._tensor))
        if show_text:
            self._text = TextInput(width=size * shape[-1] + (shape[-1] - 1) * 10)
            self._text.on_change('value', lambda a, o, n: self._update_by_text(n))
        self.update_tensor()

    def _format_text_value(self, value: float) -> str:
        return f'{value:0.2f}'

    def _create_slider(self, i: int) -> Slider:
        def update(value: float):
            self.texts[i].value = self._format_text_value(value)
            self.update_tensor()
        str_format = f'0.{"".join(["0"] * self._decimals)}'
        slider = Slider(start=self._start, end=self._end, step=self._step, value=self._tensor.view(-1)[i].item(),
                        orientation=self._slider_orientation, format=str_format, direction='ltr',
                        default_size=self._size, show_value=False)
        slider.on_change('value', lambda a, o, n: update(n))
        return slider

    def _create_text(self, i: int) -> TextInput:
        def update_slider(slider_id: int, old_value: str, value: str):
            if old_value != value:
                self.sliders[slider_id].value = float(value)

        slider = TextInput(width=self._size, margin=(0, 0, 0, 0), value=self._format_text_value(self._tensor.view(-1)[i].item()))
        slider.on_change('value', lambda a, o, n: update_slider(i, o, n))
        return slider

    @property
    def value(self) -> Tensor:
        t = self._tensor.view(-1)
        for i, s in enumerate(self.sliders):
            t[i] = s.value
        # values = [float(s.value) for s in self.sliders]
        return self._tensor
        # return torch.tensor(values)

    @value.setter
    def value(self, tensor: Tensor):
        # self._tensor = tensor
        try:
            self._internal_update = True
            for i, val in enumerate(tensor.view(-1)):
                self.sliders[i].value = val.item()
        finally:
            self._internal_update = False
            self.update_tensor()

    def update_tensor(self):
        if not self._internal_update:
            tensor = self.value
            self._plot_tensor.update(tensor)
            self.signals.on_changed.emit(tensor)

    def _update_by_text(self, value: str):
        values = [float(v.strip()) for v in re.split(r'[,\s]+', value)]
        for i, val in enumerate(values):
            # for slider, val in zip(self.sliders, values):
            self.sliders[i].value = val

    def set_text(self, text: str):
        self._text.value = text

    def _create_matrix_layout(self, arr: Union[List[Slider], List[TextInput]]):
        idx = torch.arange(0, self._tensor.numel()).view_as(self._tensor)
        rows = []
        for r in range(idx.shape[0]):
            rows.append(row([arr[idx[r, c].item()] for c in range(idx.shape[1])]))
        return column(*rows)

    def create_layout(self):
        items = []
        if self._text is not None:
            items = [self._text] + items
        if self._title is not None:
            items = [Div(text=self._title)] + items

        tab_slider = Panel(child=self._create_matrix_layout(self.sliders), title='sliders')
        tab_text = Panel(child=self._create_matrix_layout(self.texts), title='texts')
        items.append(
            row(
                column(self._plot_tensor.create_layout(), align='end'),
                Tabs(tabs=[tab_slider, tab_text])
            )
        )
        return row(
            column(*items)
        )
