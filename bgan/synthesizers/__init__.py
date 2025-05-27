"""Synthesizers module."""

from bgan.synthesizers.bgan import BGAN

__all__ = ('BGAN')


def get_all_synthesizers():
    return {name: globals()[name] for name in __all__}
