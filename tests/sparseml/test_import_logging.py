import importlib
import logging
import builtins
import sys

import pytest


def test_pytorch_base_import_failure_logs_error(monkeypatch, caplog):
    real_import = builtins.__import__

    def mocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("torch", "torchvision"):
            raise ImportError("mock fail")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mocked_import)
    sys.modules.pop("sparseml.pytorch.base", None)

    with caplog.at_level(logging.ERROR):
        mod = importlib.import_module("sparseml.pytorch.base")

    assert "Failed to import torch" in caplog.text
    assert "Failed to import torchvision" in caplog.text
    assert mod.torch_err is not None
    assert mod.torchvision_err is not None


def test_utils_init_logs_check_failure(monkeypatch, caplog):
    import sparseml.pytorch.base as base

    def raise_import(*args, **kwargs):
        raise ImportError("no torch")

    monkeypatch.setattr(base, "check_torch_install", raise_import)
    sys.modules.pop("sparseml.pytorch.utils", None)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ImportError):
            importlib.import_module("sparseml.pytorch.utils")

    assert "PyTorch installation check failed" in caplog.text

