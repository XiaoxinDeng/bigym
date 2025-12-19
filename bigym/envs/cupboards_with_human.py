"""Cupboard interaction tasks."""
from abc import ABC

import numpy as np

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet
from bigym.envs.props.human import Human
from typing import Optional
TOLERANCE = 0.1


class _HumanCupboardsInteractionEnv(BiGymEnv, ABC):
    """Base cupboards environment."""

    RESET_ROBOT_POS = np.array([-0.2, 0, 0])

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_3x1.yaml"
    _HUMAN_COUNT = 1
    _HUMAN_POS = np.array([0, 0, 0])

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
        self.humans = [Human(self._mojo) for _ in range(self._HUMAN_COUNT)]

    def _success(self) -> bool:
        for human in self.humans:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True

    def _on_reset(self, seed: Optional[int] = None):
        for human in self.humans:
            human.reset(seed=seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.

        Args:
           seed: If not None, the environment will be reset with this seed.
           options: Additional information to specify how the environment is reset
            (optional, depending on the specific environment).
        """
        self._env_health.reset()
        self._update_seed(override_seed=seed)
        self._mojo.physics.reset()
        self._action = np.zeros_like(self._action)
        self._robot.reset(self.RESET_ROBOT_POS, self.RESET_ROBOT_QUAT)
        self._on_reset(seed)
        return self.get_observation(), self.get_info()



class HumanDrawerTopOpen(_HumanCupboardsInteractionEnv):
    """Open top drawer of the cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humans:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True


class HumanDrawerTopClose(_HumanCupboardsInteractionEnv):
    """Close top drawer of the cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humans:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True

    def _on_reset(self, seed: Optional[int] = None):
        self.cabinet_drawers.set_state(np.array([0, 0, 1]))
        for human in self.humans:
            human.reset(seed=seed)


class HumanDrawersAllOpen(_HumanCupboardsInteractionEnv):
    """Open all drawers of the cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humans:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True


class HumanDrawersAllClose(_HumanCupboardsInteractionEnv):
    """Close all drawers of the cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humans:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True

    def _on_reset(self, seed: Optional[int] = None):
        self.cabinet_drawers.set_state(np.array([1, 1, 1]))
        for human in self.humans:
            human.reset(seed=seed)


class HumanWallCupboardOpen(_HumanCupboardsInteractionEnv):
    """Open doors of the wall cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humans:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True


class HumanWallCupboardClose(_HumanCupboardsInteractionEnv):
    """Close doors of the wall cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humans:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True

    def _on_reset(self, seed: Optional[int] = None):
        self.cabinet_wall.set_state(np.array([1, 1]))
        for human in self.humans:
            human.reset(seed=seed)


class HumanCupboardsOpenAll(_HumanCupboardsInteractionEnv):
    """Open all doors/drawers of the kitchen counter task."""

    def _success(self) -> bool:
        for cabinet in self.all_cabinets:
            if not np.allclose(cabinet.get_state(), 1, atol=TOLERANCE):
                return False
        for human in self.humans:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True


class HumanCupboardsCloseAll(_HumanCupboardsInteractionEnv):
    """Close all doors/drawers of the kitchen counter task."""

    def _success(self) -> bool:
        for cabinet in self.all_cabinets:
            if not np.allclose(cabinet.get_state(), 0, atol=TOLERANCE):
                return False
        for human in self.humans:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True

    def _on_reset(self, seed: Optional[int] = None):
        for cabinet in self.all_cabinets:
            cabinet.set_state(np.ones_like(cabinet.get_state()))
        for human in self.humans:
            human.reset(seed=seed)
