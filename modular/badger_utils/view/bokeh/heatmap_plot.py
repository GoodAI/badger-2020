from dataclasses import dataclass
from enum import Enum

import math
import bokeh
from bokeh.models import ColumnDataSource, ColorBar, LogTicker, LogColorMapper, LinearColorMapper, BasicTicker
from bokeh.plotting import figure
from bokeh.transform import linear_cmap, log_cmap

import pickle
from pathlib import Path

import plotly
import plotly.graph_objects as go
import plotly.express as px

from functools import partial
from pprint import pprint

import torch
from badger_utils.file.file_cache import FileCache
from badger_utils.sacred import SacredReader, SacredUtils
from badger_utils.sacred.experiment_config_diff import ExperimentConfigDiff
from badger_utils.sacred.sacred_config import SacredConfigFactory
from badger_utils.view.bokeh.slider_tensor import SliderTensor
from badger_utils.view.bokeh.slider_vector import SliderVector
from badger_utils.view.bokeh.tensor_plot import TensorPlot
from badger_utils.view.observer_utils import MultiObserver, CompoundObserver, ObserverLevel, Observer
from badger_utils.view import TensorDataMultiObserver, TensorViewer
from badger_utils.view.bokeh.bokeh_component import BokehComponent
from badger_utils.view.bokeh.bokeh_plot import BokehPlot
from badger_utils.view.bokeh_utils import plot_tensor, create_tensor_source, \
    update_figure_by_tensor, sanitize_tensor, bokeh_reg_green_cmap
import bokeh
from bokeh.io import output_notebook, show
from bokeh.layouts import row, column, layout
from bokeh.models import Div, Row, Slider, ColumnDataSource, Spacer, Button, HoverTool

from typing import List, Dict, Optional, Any, Callable, Tuple, Iterable, Union, TypeVar

from bokeh.plotting import Figure, figure
import numpy as np

from torch import Tensor


# class HeatmapPlotType(Enum):
#     LINEAR = 'linear'
#     LOG = 'log'
#

@dataclass
class HeatmapPlotConfig:
    show_text: bool = True
    title: Optional[str] = None
    cell_size: int = 25
    show_cell_grid: bool = True
    cell_grid_color: str = "#000000"
    cmap_min: Optional[float] = None
    cmap_max: Optional[float] = None
    cmap: List[str] = bokeh.palettes.Cividis256
    show_color_bar: bool = True
    plot_type: str = 'linear'
    show_xaxis: bool = True
    show_yaxis: bool = True


class HeatmapPlot(BokehComponent):
    def __init__(self, tensor: Tensor, config: HeatmapPlotConfig = None):
        self.config = config or HeatmapPlotConfig()
        t = sanitize_tensor(tensor)
        self._src = ColumnDataSource(data=create_tensor_source(t))
        self.plot = self.plot_tensor(t, self._src)

    def update(self, tensor: Tensor):
        self.update_figure_by_tensor(sanitize_tensor(tensor), self.plot)

    def create_layout(self):
        if self.config.title is not None:
            return column(
                Div(text=self._title, margin=0),
                self.plot
            )
        else:
            return self.plot

    @staticmethod
    def create_tensor_source(t: Tensor) -> Dict:
        height, width = t.size()
        xs = np.tile(np.arange(width), height)
        ys = np.arange(height - 1, -1, -1).repeat(width)
        values = t.contiguous().view(-1).detach().cpu().numpy()
        return {'x': xs + 0.5, 'y': ys + 0.5, 'row': np.flip(ys), 'column': xs, 'val': values,
                'text': [f'{i:0.1f}' for i in values],
                'text_color': ['#ffffff' if i < 0.7 else '#000000' for i in values]}

    def update_figure_by_tensor(self, t: Tensor, fig: bokeh.plotting.Figure, update_data: bool = True):
        """
        Update Figure size, axes and datasource to display passed tensor
        Args:
            t: Tensor to be shown (2D)
            fig: Figure to be updated
            update_data: When True, datasource is also updated
        """

        def to_str(a):
            return list(map(str, a))

        height, width = t.size()
        margin_x = 20 if height < 10 else 28
        margin_y = 30
        if not self.config.show_yaxis:
            margin_x -= 18
        if not self.config.show_xaxis:
            margin_y -= 18

        fig.xaxis.visible = self.config.show_xaxis
        fig.yaxis.visible = self.config.show_yaxis
        fig_width = width * self.config.cell_size + margin_x
        fig_height = height * self.config.cell_size + margin_y

        fig.x_range.factors = to_str(range(width))
        fig.y_range.factors = to_str(reversed(range(height)))
        fig.width = fig_width
        fig.height = fig_height


        if update_data:
            data = self.create_tensor_source(t)
            fig.select_one({'name': 'data_rect'}).data_source.data = data
            fig.select_one({'name': 'data_text'}).data_source.data = data

    def plot_tensor(self, t: Tensor, src: Optional[ColumnDataSource] = None,
                    text_font_size: str = '8pt') -> bokeh.plotting.Figure:
        if src is None:
            src = ColumnDataSource(data=self.create_tensor_source(t))

        tooltips = [
            ("row", "@row"),
            ("column", "@column"),
            ('value', '@val')
        ]
        fig = figure(tools="", toolbar_location=None, tooltips=tooltips, x_range=[''], y_range=[''])
        cmap_min = self.config.cmap_min if self.config.cmap_min is not None else t.min().item()
        cmap_max = self.config.cmap_max if self.config.cmap_max is not None else t.max().item()
        # color = log_cmap('val', bokeh_reg_green_cmap(), cmap_min, cmap_max)
        cmap = self.config.cmap
        # cmap = bokeh.palettes.Viridis256
        # cmap = bokeh.palettes.Cividis256
        # cmap = bokeh.palettes.Plasma256
        if self.config.plot_type == 'linear':
            color = linear_cmap('val', cmap, cmap_min, cmap_max)
            color_mapper = LinearColorMapper(palette=cmap, low=cmap_min, high=cmap_max)
            color_bar_ticker = BasicTicker()
        elif self.config.plot_type == 'log':
            color = log_cmap('val', cmap, cmap_min, cmap_max)
            color_mapper = LogColorMapper(palette=cmap, low=cmap_min, high=cmap_max)
            color_bar_ticker = LogTicker()
        else:
            raise ValueError(f'Unknown plot type: {self.config.plot_type}')

        fig.rect('x', 'y', name='data_rect', width=1, height=1, source=src,
                 fill_color=color, line_color=self.config.cell_grid_color if self.config.show_cell_grid else color,
                 line_width=0.5)
        if self.config.show_text:
            fig.text('x', 'y', name='data_text', text='text', source=src, text_color='text_color',
                     text_font_size=text_font_size,
                     x_offset=-8, y_offset=6)

        # http://docs.bokeh.org/en/latest/docs/user_guide/annotations.html#color-bars
        # color_mapper = LogColorMapper(palette="Viridis256", low=cmap_min, high=cmap_max)
        if self.config.show_color_bar:
            color_bar = ColorBar(color_mapper=color_mapper, ticker=color_bar_ticker,
                                 label_standoff=12, border_line_color=None, location=(0, 0))
            fig.add_layout(color_bar, 'right')

        self.update_figure_by_tensor(t, fig, False)
        return fig
