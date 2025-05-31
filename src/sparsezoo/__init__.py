import os
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import onnx


class File:
    def __init__(self, path: str = ""):
        self.path = path

    def download(self):
        return self

    def get_file(self, name: str):
        return File(os.path.join(self.path, name))


class _Component(File):
    @property
    def files(self):
        return [self]


class _SampleInputs:
    def download(self):
        return self

    def sample_batch(self, batch_as_list: bool = False):
        sample = [np.zeros(1)]
        return sample if batch_as_list else sample[0]


class Model:
    def __init__(self, stub: str, download_path: str | None = None):
        self.stub = stub
        self.path = download_path or stub
        self.training = _Component(self.path)
        self.onnx_model = _Component(os.path.join(self.path, "model.onnx"))
        self.deployment = SimpleNamespace(default=_Component(self.path))
        self.model_card = _Component(self.path)
        self.sample_inputs = _SampleInputs()
        self.sample_outputs = {"framework": _Component(self.path)}

    def download(self):
        return self

    def validate(self, *args, **kwargs):
        return True


# utilities module-like object
class utils:
    @staticmethod
    def validate_onnx(model):
        if isinstance(model, str):
            model = onnx.load(model)
        onnx.checker.check_model(model)
        return True

    @staticmethod
    def save_onnx(model, path):
        onnx.save(model, path)

    @staticmethod
    def load_numpy_list(path_glob):
        if isinstance(path_glob, list):
            return [np.asarray(x) for x in path_glob]
        return [np.zeros(1)]

    @staticmethod
    def download_file(url: str, save_path: str):
        open(save_path, "wb").close()


class analytics:
    class GoogleAnalytics:
        def __init__(self, *args, **kwargs):
            pass

        def send_event(self, *args, **kwargs):
            pass


# registry utilities
class RegistryMixin:
    pass

_REGISTRY = defaultdict(dict)
_ALIAS_REGISTRY = defaultdict(dict)


def standardize_lookup_name(name: str) -> str:
    return name.lower()
