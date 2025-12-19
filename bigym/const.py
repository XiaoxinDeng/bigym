"""A set of common constants."""
from __future__ import annotations

from enum import Enum, IntEnum
from pathlib import Path
import os
import numpy as np

PACKAGE_PATH = Path(__file__).parent
ASSETS_PATH = PACKAGE_PATH / "envs" / "xmls"
PRESETS_PATH = PACKAGE_PATH / "envs" / "presets"
CACHE_PATH = Path(os.environ.get("BIGYM_CACHE_PATH") or Path.home() / ".bigym")

DEMO_RELEASES = "https://github.com/chernyadev/bigym_data/releases/download"
DEMO_VERSION = "0.9.0"

WORLD_MODEL = ASSETS_PATH / "world.xml"


class ActuatorGroup(IntEnum):
    """Enum class representing the actuator group."""

    BASE = 0
    LIMB = 1
    GRIPPER = 2


class HandSide(Enum):
    """Enum class representing the hand side."""

    LEFT = "Left"
    RIGHT = "Right"


TOLERANCE_ANGULAR = np.deg2rad(2)
TOLERANCE_LINEAR = 1e-3
