# -*- coding: utf-8 -*-

"""Top-level package for bgan."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '0.11.1.dev0'

from bgan.demo import load_demo
from bgan.synthesizers.bgan import BGAN

__all__ = ('BGAN', 'load_demo')
