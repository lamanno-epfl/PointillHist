"""PointillHist: context-aware cell identity assignment for spatial transcriptomics."""

from . import datasets
from . import inference as eval
from . import plotting as pl
from . import preprocessing as pp
from . import train as tr

__version__ = "0.1.0"

__all__ = ["pp", "pl", "tr", "eval", "datasets", "__version__"]
