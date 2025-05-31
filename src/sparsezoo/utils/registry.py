from collections import defaultdict

__all__ = ["RegistryMixin", "_REGISTRY", "_ALIAS_REGISTRY", "standardize_lookup_name"]

class RegistryMixin:
    pass

_REGISTRY = defaultdict(dict)
_ALIAS_REGISTRY = defaultdict(dict)

def standardize_lookup_name(name: str) -> str:
    return name.lower()
