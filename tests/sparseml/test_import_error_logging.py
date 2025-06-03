import importlib
import builtins
import logging
import sys

import pytest


@pytest.fixture(autouse=True)
def ensure_sparseml_path():
    sys.path.insert(0, "src")
    yield
    sys.path.remove("src")


def test_onnx_base_logs_on_import_error(monkeypatch, caplog):
    import sparseml.onnx.base

    # reload module with failing imports
    original_import = builtins.__import__

    def mocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("onnx", "onnxruntime"):
            raise ImportError("mocked failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mocked_import)
    caplog.set_level(logging.ERROR)
    importlib.reload(sparseml.onnx.base)
    assert "Failed to import onnx" in caplog.text
    assert "Failed to import onnxruntime" in caplog.text
