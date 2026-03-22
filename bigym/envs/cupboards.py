"""Cupboard interaction tasks."""
from abc import ABC

import numpy as np

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet


TOLERANCE = 0.1


class _CupboardsInteractionEnv(BiGymEnv, ABC):
    """Base cupboards environment."""

    RESET_ROBOT_POS = np.array([-0.2, 0, 0])

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_3x1.yaml"

    def _initialize_env(self):
        self.cabinet_drawers = self._preset.get_props(BaseCabinet)[0]
        self.cabinet_door_left = self._preset.get_props(BaseCabinet)[1]
        self.cabinet_door_right = self._preset.get_props(BaseCabinet)[2]
        self.cabinet_wall = self._preset.get_props(WallCabinet)[0]
        self.all_cabinets = [
            self.cabinet_drawers,
            self.cabinet_door_left,
            self.cabinet_door_right,
            self.cabinet_wall,
        ]
    def get_dt(self):
            return self._mojo.model.opt.timestep * self._sub_steps_count

class DrawerTopOpen(_CupboardsInteractionEnv):
    """Open top drawer of the cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE)


class DrawerTopClose(_CupboardsInteractionEnv):
    """Close top drawer of the cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_drawers.get_state()[-1], 0, atol=TOLERANCE)

    def _on_reset(self):
        self.cabinet_drawers.set_state(np.array([0, 0, 1]))


class DrawersAllOpen(_CupboardsInteractionEnv):
    """Open all drawers of the cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_drawers.get_state(), 1, atol=TOLERANCE)


class DrawersAllClose(_CupboardsInteractionEnv):
    """Close all drawers of the cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_drawers.get_state(), 0, atol=TOLERANCE)

    def _on_reset(self):
        self.cabinet_drawers.set_state(np.array([1, 1, 1]))


class WallCupboardOpen(_CupboardsInteractionEnv):
    """Open doors of the wall cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_wall.get_state(), 1, atol=TOLERANCE)


class WallCupboardClose(_CupboardsInteractionEnv):
    """Close doors of the wall cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_wall.get_state(), 0, atol=TOLERANCE)

    def _on_reset(self):
        self.cabinet_wall.set_state(np.array([1, 1]))


class CupboardsOpenAll(_CupboardsInteractionEnv):
    """Open all doors/drawers of the kitchen counter task."""

    def _success(self) -> bool:
        for cabinet in self.all_cabinets:
            if not np.allclose(cabinet.get_state(), 1, atol=TOLERANCE):
                return False
        return True


class CupboardsCloseAll(_CupboardsInteractionEnv):
    """Close all doors/drawers of the kitchen counter task."""

    def _success(self) -> bool:
        for cabinet in self.all_cabinets:
            if not np.allclose(cabinet.get_state(), 0, atol=TOLERANCE):
                return False
        return True

    def _on_reset(self):
        for cabinet in self.all_cabinets:
            cabinet.set_state(np.ones_like(cabinet.get_state()))
