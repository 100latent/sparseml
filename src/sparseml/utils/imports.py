"""Utility helpers for optional imports."""

from types import ModuleType
import importlib

__all__ = ["optional_import", "MissingModule"]


class MissingModule(ModuleType):
    """Placeholder object for optional dependencies."""

    def __init__(self, name: str, err: Exception):
        super().__init__(name)
        self._err = err

    def __getattr__(self, item):  # pragma: no cover - simple error propagation
        raise ModuleNotFoundError(
            f"Optional dependency '{self.__name__}' is required but missing: {self._err}"
        )


def optional_import(name: str):
    """Attempt to import a module, returning a stub and error if unavailable."""
    try:
        module = importlib.import_module(name)
        return module, None
    except Exception as err:  # pragma: no cover - optional dependency stub
        return MissingModule(name, err), err
