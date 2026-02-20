"""Cupboard interaction tasks."""
from abc import ABC

import numpy as np

from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet
from bigym.envs.props.humanarm import HumanArm
from typing import Optional, Literal, Any
from bigym.action_modes import ActionMode
from bigym.utils.observation_config import ObservationConfig

TOLERANCE = 0.1


ControlMode = Literal["scripted", "position", "torque"]

class _HumanArmCupboardsInteractionEnv(BiGymEnv, ABC):
    """Base cupboards environment."""

    RESET_ROBOT_POS = np.array([-0.2, 0, 0])
    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_3x1.yaml"
    _HUMAN_COUNT = 1
    _HUMAN_POS = np.array([0, 0, 0])
    
    def __init__(self,         
                 action_mode: ActionMode,
                 observation_config: ObservationConfig = ObservationConfig(),
                 render_mode = None, 
                 start_seed = None, 
                 control_frequency = CONTROL_FREQUENCY_MAX, 
                 robot_cls = None, 
                 arm_action_mode:ControlMode="scripted"):
        self.arm_action_mode : ControlMode = arm_action_mode
        super().__init__(action_mode, 
                         observation_config, 
                         render_mode, 
                         start_seed, 
                         control_frequency, 
                         robot_cls)

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
        self.humanarms : list[HumanArm] = [HumanArm(self._mojo, kinematic=True) for _ in range(self._HUMAN_COUNT)]
        for human in self.humanarms:
            human.set_mode(self.arm_action_mode)
        self.arm_cmd = np.zeros(3) # radians

    def _success(self) -> bool:
        for human in self.humanarms:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True
    
    def get_dt(self):
        return self._mojo.model.opt.timestep * self._sub_steps_count

    def step(
        self, action: np.ndarray, fast: bool = False, arm_action:np.ndarray=None
    ) -> tuple[Any, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action: Action to take.
            fast: If True, perform the environment step without processing observations
                and return default values. Useful when performance is crucial,
                but observations are not required, e.g., demo collection in VR.

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        self._step_cache.clean()
        self._step_mujoco_simulation(action)
        self._on_step(arm_action)
        self._action = action
        if fast:
            return {}, 0, False, False, {}
        else:
            return (
                self.get_observation(),
                self.reward,
                self.terminate,
                self.truncate,
                self.get_info(),
            )

    def _get_task_privileged_obs(self) -> dict[str, Any]:
        return {
            "human_arm_qpos": np.concatenate([human.get_state()['qpos'] for human in self.humanarms], dtype=np.float32),
            "human_arm_qvel": np.concatenate([human.get_state()['qvel'] for human in self.humanarms], dtype=np.float32),
        }
        

    def _on_step(self, arm_action:np.ndarray=None):
        human: HumanArm
        for human in self.humanarms:
            if arm_action is not None:
                human.set_qpos_target(arm_action)
            human._on_step(self.get_dt())

    def _on_reset(self, seed: Optional[int] = None):
        for human in self.humanarms:
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



class HumanArmDrawerTopOpen(_HumanArmCupboardsInteractionEnv):
    """Open top drawer of the cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humanarms:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True


class HumanArmDrawerTopClose(_HumanArmCupboardsInteractionEnv):
    """Close top drawer of the cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humanarms:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True

    def _on_reset(self, seed: Optional[int] = None):
        self.cabinet_drawers.set_state(np.array([0, 0, 1]))
        for human in self.humanarms:
            human.reset(seed=seed)


class HumanArmDrawersAllOpen(_HumanArmCupboardsInteractionEnv):
    """Open all drawers of the cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humanarms:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True


class HumanArmDrawersAllClose(_HumanArmCupboardsInteractionEnv):
    """Close all drawers of the cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humanarms:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True

    def _on_reset(self, seed: Optional[int] = None):
        self.cabinet_drawers.set_state(np.array([1, 1, 1]))
        for human in self.humanarms:
            human.reset(seed=seed)


class HumanArmWallCupboardOpen(_HumanArmCupboardsInteractionEnv):
    """Open doors of the wall cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humanarms:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True


class HumanArmWallCupboardClose(_HumanArmCupboardsInteractionEnv):
    """Close doors of the wall cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        for human in self.humanarms:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True

    def _on_reset(self, seed: Optional[int] = None):
        self.cabinet_wall.set_state(np.array([1, 1]))
        for human in self.humanarms:
            human.reset(seed=seed)


class HumanArmCupboardsOpenAll(_HumanArmCupboardsInteractionEnv):
    """Open all doors/drawers of the kitchen counter task."""

    def _success(self) -> bool:
        for cabinet in self.all_cabinets:
            if not np.allclose(cabinet.get_state(), 1, atol=TOLERANCE):
                return False
        for human in self.humanarms:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True


class HumanArmCupboardsCloseAll(_HumanArmCupboardsInteractionEnv):
    """Close all doors/drawers of the kitchen counter task."""

    def _success(self) -> bool:
        for cabinet in self.all_cabinets:
            if not np.allclose(cabinet.get_state(), 0, atol=TOLERANCE):
                return False
        for human in self.humanarms:
            if not human.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
        return True

    def _on_reset(self, seed: Optional[int] = None):
        for cabinet in self.all_cabinets:
            cabinet.set_state(np.ones_like(cabinet.get_state()))
        for human in self.humanarms:
            human.reset(seed=seed)
