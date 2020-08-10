from dataclasses import dataclass, field, asdict

import itertools
from typing import Optional, Callable, Any, Tuple, Dict, List, Union

from bokeh.core.enums import DashPattern

from badger_utils.sacred.sacred_config import SacredConfig
from badger_utils.sacred import SacredReader, SacredUtils
from badger_utils.view.bokeh.bokeh_component import BokehComponent
import pandas as pd
from badger_utils.view.signals import Signal1, signal
from bokeh.events import DoubleTap
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Slider, CheckboxGroup
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import Figure


@dataclass
class LineStyle:
    line_width: int = 1
    line_dash: Union[str, List[int]] = 'solid'
    line_color: str = None


@dataclass
class LinePlotConfig:
    width: int = 1000
    height: int = 300
    y_axis_type: str = "log"
    title: str = 'line'
    x_label: str = 'epoch'
    x_range: Tuple[Optional[int], Optional[int]] = (None, None)
    y_range: Tuple[Optional[int], Optional[int]] = (None, None)
    # line_width: int = 1
    lines_config: Dict[str, LineStyle] = field(default_factory=dict)


class LinePlotSignals:
    def __init__(self):
        self.on_epoch_selected = signal(int)


class LinePlot(BokehComponent):
    def __init__(self, config: LinePlotConfig, df: pd.DataFrame):
        self.config = config
        self.df = df
        self.signals = LinePlotSignals()
        self.ds = ColumnDataSource(df)

    def update(self, df: pd.DataFrame):
        self.df = df
        # self.df.columns = ['line']
        self.ds.data = df

    def _create_line_figure(self, df: pd.DataFrame):
        colors = itertools.cycle(palette)
        df.columns = [str(i) for i in df.columns]
        # self.ds = ColumnDataSource(df)
        c = self.config
        fig = Figure(y_axis_type=c.y_axis_type, width=c.width, height=c.height, title=c.title, x_axis_label=c.x_label)
        fig.below[0].formatter.use_scientific = False
        if c.x_range[0] is not None:
            fig.x_range.start = c.x_range[0]
        if c.x_range[1] is not None:
            fig.x_range.end = c.x_range[1]
        if c.y_range[0] is not None:
            fig.y_range.start = c.y_range[0]
        if c.y_range[1] is not None:
            fig.y_range.end = c.y_range[1]

        for column, color in zip(df.columns, colors):
            params = asdict(c.lines_config[column]) if column in c.lines_config else {}
            if 'line_color' not in params or params['line_color'] is None:
                params['line_color'] = color
            glyph = fig.line(x='index', y=column, source=self.ds, color=color, legend_label=column, **params)
            fig.add_tools(
                # HoverTool(tooltips=[(c.x_label, "@index")] + [(f"{column}", f"@{column}")],
                HoverTool(tooltips=[(f"{column}", f"@{column}")], mode='vline', renderers=[glyph]))

        fig.on_event(DoubleTap, lambda e: self.signals.on_epoch_selected.emit(int(e.x)))
        fig.legend.location = 'bottom_left'
        fig.legend.click_policy = "hide"
        fig.legend.label_text_font_size = '12px'
        fig.legend.spacing = -4
        fig.legend.background_fill_alpha = 0.6
        return fig

    def create_layout(self):
        return self._create_line_figure(self.df)
