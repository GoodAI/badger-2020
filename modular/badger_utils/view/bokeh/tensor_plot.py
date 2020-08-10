from badger_utils.view.bokeh_utils import create_tensor_source, plot_tensor, update_figure_by_tensor, sanitize_tensor
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div
from torch import Tensor
from typing import Optional

from badger_utils.view.bokeh.bokeh_component import BokehComponent


class TensorPlot(BokehComponent):
    def __init__(self, tensor: Tensor, title: Optional[str] = None):
        t = sanitize_tensor(tensor)
        self._src = ColumnDataSource(data=create_tensor_source(t))
        self.plot = plot_tensor(t, self._src)
        self._title = title

    def update(self, tensor: Tensor):
        update_figure_by_tensor(sanitize_tensor(tensor), self.plot)

    def create_layout(self):
        if self._title is not None:
            return column(
                Div(text=self._title, margin=0),
                self.plot
            )
        else:
            return self.plot
