from dataclasses import dataclass

import itertools
from typing import Optional, Callable, Any

from badger_utils.sacred.sacred_config import SacredConfig
from badger_utils.sacred import SacredReader, SacredUtils
from badger_utils.view.bokeh.bokeh_component import BokehComponent
import pandas as pd

from badger_utils.view.bokeh.bokeh_plot import BokehPlot
from badger_utils.view.signals import Signal1, signal
from bokeh.events import DoubleTap
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Slider, CheckboxGroup, Select
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import Figure


@dataclass
class MetricsPlotConfig:
    width: int = 1000
    height: int = 300
    y_axis_type: str = "log"
    title: str = 'Metrics'
    x_label: str = 'epoch'
    # x_range: Tuple[int, int] = (-2, 2)
    # y_range: Tuple[int, int] = (-2, 2)


class MetricsPlotSignals:
    def __init__(self):
        self.on_epoch_selected = signal(int)
        self.on_double_tap = signal(float, float)


class ExperimentMetricsPlot(BokehComponent):

    def __init__(self, config: MetricsPlotConfig, sacred_config: SacredConfig, experiment_id: int,
                 metric_name_filter: Callable[[str], bool] = None):
        self._sacred_utils = SacredUtils(sacred_config)
        self._metric_name_filter = metric_name_filter
        self._experiment_id = experiment_id
        self._widget_yscale_select = Select(title='yscale', options=['log', 'linear'], width=80)
        self._widget_yscale_select.on_change('value', lambda a, o, n: self._update_plot())
        self._widget_smooth_slider = Slider(start=5, end=250, step=5, value=20, title='Smoothing window', align='start')
        self._widget_smooth_slider.on_change('value', lambda a, o, n: self._update_plot())
        self._widget_smooth_checkbox = CheckboxGroup(labels=["Smooth"], active=[0], align='end', width_policy='min')
        self._widget_smooth_checkbox.on_click(lambda _: self._smooth_update())

        self._widget_plot_pane = row()
        self._config = config
        self.signals = MetricsPlotSignals()

    @property
    def experiment_id(self) -> int:
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, experiment_id: int):
        self._experiment_id = experiment_id
        self._load_metrics()
        self._update_plot()

    def create_layout(self):
        return column(
            self._widget_plot_pane,
            # self._metric_plot.create_layout(),
            row(
                self._widget_smooth_checkbox,
                self._widget_smooth_slider,
                self._widget_yscale_select
            )
        )

    def _update_plot(self):
        self._widget_plot_pane.children.clear()
        c = self._config
        y_axis_type = self._widget_yscale_select.value or 'log'
        fig = BokehPlot.plot_df(self._processed_metrics(), width=c.width, height=c.height, title=c.title,
                                x_label=c.x_label, y_axis_type=y_axis_type)

        def double_tap(e):
            self.signals.on_epoch_selected.emit(int(e.x))
            self.signals.on_double_tap.emit(e.x, e.y)

        fig.on_event(DoubleTap, double_tap)
        self._widget_plot_pane.children.append(fig)

    def _load_metrics(self):
        self.metrics = self._sacred_utils.get_reader(self._experiment_id).load_metrics()
        if self._metric_name_filter is not None:
            columns = [c for c in self.metrics.columns if self._metric_name_filter(c)]
            self.metrics = self.metrics[columns]

    def _processed_metrics(self) -> pd.DataFrame:
        if self._smooth_active():
            # rolling_window = int(len(self.metrics) * self._widget_smooth_slider.value * 0.2)
            rolling_window = int(self._widget_smooth_slider.value)
            return self.metrics.rolling(window=rolling_window, center=True).mean()
        else:
            return self.metrics

    def _smooth_active(self) -> bool:
        return 0 in self._widget_smooth_checkbox.active

    def _smooth_update(self):
        self._widget_smooth_slider.disabled = not self._smooth_active()
        self._widget_smooth_slider.visible = self._smooth_active()
        self._update_plot()
