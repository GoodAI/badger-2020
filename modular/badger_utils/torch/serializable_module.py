from typing import Any, Dict

from badger_utils.torch.serializable import Serializable
from badger_utils.torch.device_aware import DeviceAwareModule


class SerializableModule(DeviceAwareModule, Serializable):
    def serialize(self) -> Dict[str, Any]:
        return {
            'model': self.state_dict()
        }

    def deserialize(self, data: Dict[str, Any], strict: bool = True):
        self.load_state_dict(data['model'], strict)
