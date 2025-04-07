"""Synthesizers module."""

from bgan.synthesizers.bgan import BGAN
from bgan.synthesizers.tvae import TVAE

__all__ = ('BGAN', 'TVAE')


def get_all_synthesizers():
    return {name: globals()[name] for name in __all__}
