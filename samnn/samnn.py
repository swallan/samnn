from Network import Network
from Layer import Layer
__all__ = ['Layer', 'Network']

def __dir__():
    return __all__

def __getattr__(name):
    if name in __all__:
        import importlib as _importlib
        return _importlib.import_module(f'samnn.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'samnn' has no attribute '{name}'"
            )