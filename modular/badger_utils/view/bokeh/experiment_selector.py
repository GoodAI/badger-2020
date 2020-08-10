import traceback

from badger_utils.sacred import SacredUtils, SacredConfig
from badger_utils.view.bokeh.bokeh_component import BokehComponent
from badger_utils.view.signals import signal
from bokeh.layouts import row
from bokeh.models import Div, Button, TextInput


class ExperimentSelectorSignals:
    def __init__(self):
        self.on_experiment_selected = signal(int)


class ExperimentSelector(BokehComponent):
    _experiment_id = 0

    def __init__(self, sacred_config: SacredConfig):
        self.sacred_utils = SacredUtils(sacred_config)
        self.signals = ExperimentSelectorSignals()

        self._widget_text_experiment_id = TextInput(value='170', title="Expermient ID", align="end")
        self._widget_text_experiment_id.on_change('value', lambda a, o, n: self._on_update_experiment_id())
        self._widget_button_last_run = Button(label="Last run", align="end")
        self._widget_button_last_run.on_click(self._on_last_run)
        self._widget_button_reload = Button(label="Read experiment", align="end")
        self._widget_button_reload.on_click(self._on_update_experiment_id)

    @property
    def experiment_id(self):
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, value):
        self._experiment_id = value
        self._widget_text_experiment_id.value = str(self._experiment_id)
        self._widget_button_reload.label = f'Experiment Id: {self._experiment_id}'
        self.signals.on_experiment_selected.emit(self._experiment_id)

    def select_last_experiment(self):
        self._on_last_run()

    def _on_last_run(self):
        self.experiment_id = self.sacred_utils.get_last_run().id

    def _on_update_experiment_id(self):
        try:
            self.experiment_id = int(self._widget_text_experiment_id.value)
        except Exception as e:
            traceback.print_exc()

    def create_layout(self):
        return row(self._widget_text_experiment_id, self._widget_button_reload, self._widget_button_last_run)
