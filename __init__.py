"""Chaos Intelligence

A Layer API built on top of `tensorflow.keras` Functional API.

The API is recommended to be used on data that appears to be random but have an underlying pattern.
Such data is said to be in deterministic chaos, and is sensitive to initial conditions.
This API allows the changing of applied functionality of layer, when triggered by deterministic function.

"""

from src.core import *
from src.DeterministicFunctions import *
from src.OverlayLayer import *

module_version = "0.0.0d"
module_status = "development"

