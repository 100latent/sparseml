import numpy as np
import onnx
from onnx import ModelProto
from typing import Any, Iterable

__all__ = [
    "validate_onnx",
    "save_onnx",
    "load_model",
    "onnx_includes_external_data",
    "load_numpy_list",
    "download_file",
    "EXTERNAL_ONNX_DATA_NAME",
]

EXTERNAL_ONNX_DATA_NAME = "model.data"

def validate_onnx(model: Any) -> bool:
    if isinstance(model, str):
        model = onnx.load(model)
    onnx.checker.check_model(model)
    return True

def save_onnx(model: ModelProto, path: str, **kwargs) -> None:
    onnx.save(model, path)

def load_model(model: Any, load_external_data: bool = True) -> ModelProto:
    if isinstance(model, ModelProto):
        return model
    return onnx.load(model)

def onnx_includes_external_data(model: ModelProto) -> bool:
    if isinstance(model, str):
        model = onnx.load(model)
    return any(init.external_data for init in model.graph.initializer)

def load_numpy_list(path_glob: Iterable[Any]):
    if isinstance(path_glob, list):
        return [np.asarray(x) for x in path_glob]
    return [np.zeros(1)]

def download_file(url: str, save_path: str) -> None:
    open(save_path, "wb").close()
