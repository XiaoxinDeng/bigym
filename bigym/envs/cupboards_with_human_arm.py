"""Cupboard interaction tasks."""
from abc import ABC
from typing import Optional, Literal, Any, Sequence

import mujoco
import numpy as np
from gymnasium import spaces

from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet
from bigym.envs.props.humanarm import HumanArm
from bigym.action_modes import ActionMode
from bigym.utils.observation_config import ObservationConfig

TOLERANCE = 0.1


ControlMode = Literal["scripted", "position", "torque"]


DEFAULT_BLOCKER_HUMAN_JOINT_NAMES = [
    "cylinder_arm/arm_tx",
    "cylinder_arm/arm_ty",
    "cylinder_arm/arm_shoulder_base",
    "cylinder_arm/arm_shoulder_yaw",
    "cylinder_arm/arm_shoulder_pitch",
    "cylinder_arm/arm_elbow",
]

DEFAULT_BLOCKER_Q_OUTSIDE = {
    "arm_tx": -0.85,
    "arm_ty": -0.31,
    "arm_shoulder_base": 0.0,
    "arm_shoulder_yaw": 0.0,
    "arm_shoulder_pitch": -0.35,
    "arm_elbow": 1.20,
}

DEFAULT_BLOCKER_Q_BLOCK = {
    "arm_tx": 0.55,
    "arm_ty": 0.31,
    "arm_shoulder_base": 0.0,
    "arm_shoulder_yaw": 1.57,
    "arm_shoulder_pitch": -0.45,
    "arm_elbow": 1.00,
}


def _short_mujoco_name(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _resolve_mujoco_name(model, obj_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, obj_type, name)
    if object_id >= 0:
        return int(object_id)

    matches = []
    if obj_type == mujoco.mjtObj.mjOBJ_JOINT:
        count = model.njnt
    elif obj_type == mujoco.mjtObj.mjOBJ_SITE:
        count = model.nsite
    elif obj_type == mujoco.mjtObj.mjOBJ_GEOM:
        count = model.ngeom
    else:
        count = 0

    for i in range(count):
        candidate = mujoco.mj_id2name(model, obj_type, i)
        if candidate == name or candidate.endswith(f"/{name}"):
            matches.append(i)

    if len(matches) == 1:
        return int(matches[0])
    if len(matches) > 1:
        raise ValueError(f"MuJoCo name {name!r} is ambiguous: {matches}")
    raise ValueError(f"MuJoCo name {name!r} was not found")


def _default_blocker_keyframe(
    human_joint_names: Sequence[str],
    values_by_short_name: dict[str, float],
) -> list[float]:
    values = []
    for name in human_joint_names:
        short_name = _short_mujoco_name(str(name))
        if short_name not in values_by_short_name:
            raise ValueError(
                f"No default temporary human blocker q value for joint {name!r}. "
                "Provide q_outside/q_block explicitly for custom joints."
            )
        values.append(float(values_by_short_name[short_name]))
    return values


class TemporaryDrawerArmBlocker:
    """Deterministic human forearm blocker for drawer-reaching benchmarks."""

    PHASE_OUTSIDE = "outside"
    PHASE_ENTER = "enter"
    PHASE_HOLD = "hold"
    PHASE_EXIT = "exit"
    PHASE_DONE = "done"

    def __init__(
        self,
        model,
        data,
        human_joint_names,
        ee_site_name,
        handle_site_name,
        q_outside,
        q_block,
        q_exit=None,
        trigger_dist=0.25,
        enter_duration=1.2,
        hold_duration=1.2,
        exit_duration=1.2,
        natural_motion_scale=1.0,
        exit_after_y_peak: bool = False,
        max_joint_speed: Optional[float] = None,
    ):
        self.model = model
        self.data = data
        self.human_joint_names = [str(name) for name in human_joint_names]
        self.ee_site_name = str(ee_site_name)
        self.handle_site_name = str(handle_site_name)
        self.trigger_dist = float(trigger_dist)
        self.enter_duration = float(enter_duration)
        self.hold_duration = float(hold_duration)
        self.exit_duration = float(exit_duration)
        self.natural_motion_scale = max(0.0, float(natural_motion_scale))
        self.exit_after_y_peak = bool(exit_after_y_peak)
        self.max_joint_speed = (
            None if max_joint_speed is None else max(0.0, float(max_joint_speed))
        )

        self.q_outside = self._as_keyframe("q_outside", q_outside)
        self.q_block = self._as_keyframe("q_block", q_block)
        self.q_exit = self.q_outside.copy() if q_exit is None else self._as_keyframe("q_exit", q_exit)

        self._joint_ids = np.array(
            [
                _resolve_mujoco_name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
                for name in self.human_joint_names
            ],
            dtype=np.int32,
        )
        self._qpos_adr = np.asarray(self.model.jnt_qposadr[self._joint_ids], dtype=np.int32)
        self._qvel_adr = np.asarray(self.model.jnt_dofadr[self._joint_ids], dtype=np.int32)
        self._joint_index_by_short_name = {
            _short_mujoco_name(name): i
            for i, name in enumerate(self.human_joint_names)
        }
        self._joint_ranges = np.asarray(self.model.jnt_range[self._joint_ids], dtype=np.float64)
        self._joint_limited = np.asarray(self.model.jnt_limited[self._joint_ids], dtype=bool)
        self._root_joint_id = self._find_root_free_joint_id()
        if self._root_joint_id is None:
            self._root_qpos_adr = None
            self._root_qvel_adr = None
            self._root_qpos = None
        else:
            self._root_qpos_adr = int(self.model.jnt_qposadr[self._root_joint_id])
            self._root_qvel_adr = int(self.model.jnt_dofadr[self._root_joint_id])
            self._root_qpos = np.asarray(
                self.data.qpos[self._root_qpos_adr:self._root_qpos_adr + 7],
                dtype=np.float64,
            ).copy()
        self._actuator_ids = self._find_human_actuator_ids()

        unsupported = [
            name
            for name, joint_id in zip(self.human_joint_names, self._joint_ids)
            if int(self.model.jnt_type[joint_id])
            not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        ]
        if unsupported:
            raise ValueError(
                "TemporaryDrawerArmBlocker only supports scalar hinge/slide joints; "
                f"unsupported joints: {unsupported}"
            )

        self._ee_site_id = _resolve_mujoco_name(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.ee_site_name,
        )
        self._handle_site_id = _resolve_mujoco_name(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.handle_site_name,
        )

        self._human_geom_ids = self._find_human_geom_ids()
        self._robot_geom_ids = self._find_robot_geom_ids()

        self.phase = self.PHASE_OUTSIDE
        self.triggered = False
        self.time_in_phase = 0.0
        self._phase_start_q = self.q_outside.copy()
        self._phase_target_q = self.q_outside.copy()
        self._last_q = self.q_outside.copy()
        self.last_info = {}
        self.reset()

    def _as_keyframe(self, name: str, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape != (len(self.human_joint_names),):
            raise ValueError(
                f"{name} must have {len(self.human_joint_names)} values for "
                f"human_joint_names={self.human_joint_names}, got shape {q.shape}"
            )
        return q

    def _find_human_geom_ids(self) -> np.ndarray:
        ids = []
        tokens = ("forearm_geom", "upperarm_geom")
        for geom_id in range(int(self.model.ngeom)):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if any(token in name for token in tokens):
                ids.append(geom_id)
        return np.asarray(ids, dtype=np.int32)

    def _find_robot_geom_ids(self) -> np.ndarray:
        ids = []
        for geom_id in range(int(self.model.ngeom)):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if not name.startswith("h1/"):
                continue
            if int(self.model.geom_contype[geom_id]) == 0 and int(self.model.geom_conaffinity[geom_id]) == 0:
                continue
            ids.append(geom_id)
        return np.asarray(ids, dtype=np.int32)

    def _find_root_free_joint_id(self) -> Optional[int]:
        if len(self._joint_ids) == 0:
            return None

        body_id = int(self.model.jnt_bodyid[int(self._joint_ids[0])])
        while body_id > 0:
            joint_matches = [
                joint_id
                for joint_id in range(int(self.model.njnt))
                if int(self.model.jnt_bodyid[joint_id]) == body_id
                and int(self.model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
            ]
            if joint_matches:
                return int(joint_matches[0])
            body_id = int(self.model.body_parentid[body_id])
        return None

    def _find_human_actuator_ids(self) -> np.ndarray:
        joint_ids = set(int(joint_id) for joint_id in self._joint_ids)
        ids = []
        for actuator_id in range(int(self.model.nu)):
            actuator_joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if actuator_joint_id in joint_ids:
                ids.append(actuator_id)
        return np.asarray(ids, dtype=np.int32)

    def reset(self):
        self.phase = self.PHASE_OUTSIDE
        self.triggered = False
        self.time_in_phase = 0.0
        self._phase_start_q = self.q_outside.copy()
        self._phase_target_q = self.q_outside.copy()
        self._apply_q(self.q_outside, dt=None)
        self.last_info = self._make_info()
        return dict(self.last_info)

    def update(self, dt):
        dt = max(0.0, float(dt))
        ee_to_handle_dist = self._ee_to_handle_dist()

        if (
            self.phase == self.PHASE_OUTSIDE
            and not self.triggered
            and ee_to_handle_dist < self.trigger_dist
        ):
            self.triggered = True
            self._start_phase(self.PHASE_ENTER, self.q_outside, self.q_block)

        if self.phase == self.PHASE_ENTER:
            next_phase = self.PHASE_EXIT if self.exit_after_y_peak else self.PHASE_HOLD
            next_target = self.q_exit if self.exit_after_y_peak else self.q_block
            self._advance_interpolated_phase(
                dt,
                self.enter_duration,
                next_phase=next_phase,
                next_start=self.q_block,
                next_target=next_target,
            )
        elif self.phase == self.PHASE_HOLD:
            self.time_in_phase += dt
            tau = 1.0 if self.hold_duration <= 0.0 else np.clip(
                self.time_in_phase / self.hold_duration,
                0.0,
                1.0,
            )
            q = self._with_natural_joint_motion(self.q_block, self.PHASE_HOLD, tau)
            self._apply_q(q, dt=dt)
            if self.time_in_phase >= self.hold_duration:
                self._start_phase(self.PHASE_EXIT, self.q_block, self.q_exit)
        elif self.phase == self.PHASE_EXIT:
            self._advance_interpolated_phase(
                dt,
                self.exit_duration,
                next_phase=self.PHASE_DONE,
                next_start=self.q_exit,
                next_target=self.q_exit,
            )
        elif self.phase == self.PHASE_OUTSIDE:
            self._apply_q(self.q_outside, dt=dt)
        elif self.phase == self.PHASE_DONE:
            self.time_in_phase += dt
            self._apply_q(self.q_exit, dt=dt)
        else:
            raise RuntimeError(f"Unknown temporary human blocker phase: {self.phase}")

        self.last_info = self._make_info(ee_to_handle_dist=ee_to_handle_dist)
        return dict(self.last_info)

    def _start_phase(self, phase: str, start_q: np.ndarray, target_q: np.ndarray):
        self.phase = phase
        self.time_in_phase = 0.0
        self._phase_start_q = np.asarray(start_q, dtype=np.float64).copy()
        self._phase_target_q = np.asarray(target_q, dtype=np.float64).copy()

    def _advance_interpolated_phase(
        self,
        dt: float,
        duration: float,
        next_phase: str,
        next_start: np.ndarray,
        next_target: np.ndarray,
    ):
        self.time_in_phase += dt
        duration = self._speed_limited_duration(
            duration,
            self._phase_start_q,
            self._phase_target_q,
        )
        if duration <= 0.0:
            tau = 1.0
        else:
            tau = np.clip(self.time_in_phase / duration, 0.0, 1.0)
        s = self._smoothstep(tau)
        q = self._phase_start_q + s * (self._phase_target_q - self._phase_start_q)
        q = self._with_natural_joint_motion(q, self.phase, tau)
        self._apply_q(q, dt=dt)

        if tau >= 1.0:
            self._apply_q(self._phase_target_q, dt=dt)
            self._start_phase(next_phase, next_start, next_target)

    def _smoothstep(self, tau: float) -> float:
        tau = float(np.clip(tau, 0.0, 1.0))
        return 3.0 * tau * tau - 2.0 * tau * tau * tau

    def _speed_limited_duration(
        self,
        requested_duration: float,
        start_q: np.ndarray,
        target_q: np.ndarray,
    ) -> float:
        duration = max(0.0, float(requested_duration))
        if self.max_joint_speed is None or self.max_joint_speed <= 0.0:
            return duration
        max_delta = float(
            np.max(
                np.abs(
                    np.asarray(target_q, dtype=np.float64)
                    - np.asarray(start_q, dtype=np.float64)
                )
            )
        )
        if max_delta <= 0.0:
            return duration
        # Smoothstep reaches 1.5x the average slope at mid-transition.
        min_duration = 1.5 * max_delta / max(self.max_joint_speed, 1e-9)
        return max(duration, min_duration)

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).copy()
        lo = self._joint_ranges[:, 0]
        hi = self._joint_ranges[:, 1]
        q[self._joint_limited] = np.clip(
            q[self._joint_limited],
            lo[self._joint_limited],
            hi[self._joint_limited],
        )
        return q

    def _with_natural_joint_motion(self, q: np.ndarray, phase: str, tau: float) -> np.ndarray:
        if self.natural_motion_scale <= 0.0:
            return np.asarray(q, dtype=np.float64)

        tau = float(np.clip(tau, 0.0, 1.0))
        q = np.asarray(q, dtype=np.float64).copy()
        envelope = np.sin(np.pi * tau)

        if phase == self.PHASE_ENTER:
            offsets = {
                "arm_shoulder_base": 0.05 * envelope,
                "arm_shoulder_yaw": 0.03 * envelope,
                "arm_shoulder_pitch": -0.08 * envelope,
                "arm_elbow": 0.10 * envelope,
            }
        elif phase == self.PHASE_HOLD:
            sway = envelope * np.sin(2.0 * np.pi * tau)
            settle = envelope * np.sin(2.0 * np.pi * tau + 0.7)
            offsets = {
                "arm_shoulder_base": 0.025 * sway,
                "arm_shoulder_yaw": 0.035 * sway,
                "arm_shoulder_pitch": -0.035 * settle,
                "arm_elbow": 0.045 * settle,
            }
        elif phase == self.PHASE_EXIT:
            offsets = {
                "arm_shoulder_base": -0.04 * envelope,
                "arm_shoulder_yaw": -0.025 * envelope,
                "arm_shoulder_pitch": -0.06 * envelope,
                "arm_elbow": 0.08 * envelope,
            }
        else:
            return q

        for joint_name, offset in offsets.items():
            joint_index = self._joint_index_by_short_name.get(joint_name)
            if joint_index is not None:
                q[joint_index] += self.natural_motion_scale * offset
        return self._clip_q(q)

    def _apply_q(self, q, dt=None):
        q = self._clip_q(np.asarray(q, dtype=np.float64).reshape(-1))
        if self._root_qpos_adr is not None:
            self.data.qpos[self._root_qpos_adr:self._root_qpos_adr + 7] = self._root_qpos
            self.data.qvel[self._root_qvel_adr:self._root_qvel_adr + 6] = 0.0
        self.data.qpos[self._qpos_adr] = q
        self.data.qvel[self._qvel_adr] = 0.0
        if self._actuator_ids.size:
            self.data.ctrl[self._actuator_ids] = 0.0
        self._last_q = q.copy()
        mujoco.mj_forward(self.model, self.data)

    def _ee_to_handle_dist(self) -> float:
        ee_pos = np.asarray(self.data.site_xpos[self._ee_site_id], dtype=np.float64)
        handle_pos = np.asarray(self.data.site_xpos[self._handle_site_id], dtype=np.float64)
        return float(np.linalg.norm(ee_pos - handle_pos))

    def _min_robot_human_distance(self):
        if self._human_geom_ids.size == 0 or self._robot_geom_ids.size == 0:
            return None
        human_pos = np.asarray(self.data.geom_xpos[self._human_geom_ids], dtype=np.float64)
        robot_pos = np.asarray(self.data.geom_xpos[self._robot_geom_ids], dtype=np.float64)
        human_r = np.asarray(self.model.geom_rbound[self._human_geom_ids], dtype=np.float64)
        robot_r = np.asarray(self.model.geom_rbound[self._robot_geom_ids], dtype=np.float64)
        delta = human_pos[:, None, :] - robot_pos[None, :, :]
        center_dist = np.linalg.norm(delta, axis=-1)
        clearance = center_dist - human_r[:, None] - robot_r[None, :]
        return float(np.min(clearance))

    def _make_info(self, ee_to_handle_dist=None) -> dict[str, Any]:
        if ee_to_handle_dist is None:
            ee_to_handle_dist = self._ee_to_handle_dist()
        min_robot_human_distance = self._min_robot_human_distance()
        return {
            "human_phase": self.phase,
            "ee_to_handle_dist": float(ee_to_handle_dist),
            "human_blocker_triggered": bool(self.triggered),
            "human_time_in_phase": float(self.time_in_phase),
            "min_robot_human_distance": min_robot_human_distance,
            "human_min_robot_distance": min_robot_human_distance,
            "human_exit_after_y_peak": bool(self.exit_after_y_peak),
            "human_max_joint_speed": self.max_joint_speed,
        }

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
                 arm_action_mode:ControlMode="scripted",
                 enable_temporary_human_blocker: bool = False,
                 trigger_dist: float = 0.25,
                 enter_duration: float = 1.2,
                 hold_duration: float = 1.2,
                 exit_duration: float = 1.2,
                 natural_motion_scale: float = 1.0,
                 exit_after_y_peak: bool = False,
                 max_blocker_joint_speed: Optional[float] = None,
                 q_outside: Optional[Sequence[float]] = None,
                 q_block: Optional[Sequence[float]] = None,
                 q_exit: Optional[Sequence[float]] = None,
                 human_joint_names: Optional[Sequence[str]] = None,
                 ee_site_name: str = "right_end_effector",
                 handle_site_name: str = "drawer_small_4"):
        self.arm_action_mode : ControlMode = arm_action_mode
        self.enable_temporary_human_blocker = bool(enable_temporary_human_blocker)
        self.temporary_human_blocker_config = {
            "trigger_dist": trigger_dist,
            "enter_duration": enter_duration,
            "hold_duration": hold_duration,
            "exit_duration": exit_duration,
            "natural_motion_scale": natural_motion_scale,
            "exit_after_y_peak": exit_after_y_peak,
            "max_blocker_joint_speed": max_blocker_joint_speed,
            "q_outside": q_outside,
            "q_block": q_block,
            "q_exit": q_exit,
            "human_joint_names": human_joint_names,
            "ee_site_name": ee_site_name,
            "handle_site_name": handle_site_name,
        }
        self._temporary_human_blocker: Optional[TemporaryDrawerArmBlocker] = None
        self._temporary_human_blocker_info: dict[str, Any] = {}
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
        self._initialize_temporary_human_blocker()

    def _initialize_temporary_human_blocker(self):
        if not self.enable_temporary_human_blocker:
            self._temporary_human_blocker = None
            self._temporary_human_blocker_info = {}
            return

        config = dict(self.temporary_human_blocker_config)
        human_joint_names = config["human_joint_names"]
        if human_joint_names is None:
            human_joint_names = list(DEFAULT_BLOCKER_HUMAN_JOINT_NAMES)
        else:
            human_joint_names = [str(name) for name in human_joint_names]

        q_outside = config["q_outside"]
        if q_outside is None:
            q_outside = _default_blocker_keyframe(
                human_joint_names,
                DEFAULT_BLOCKER_Q_OUTSIDE,
            )
        q_block = config["q_block"]
        if q_block is None:
            q_block = _default_blocker_keyframe(
                human_joint_names,
                DEFAULT_BLOCKER_Q_BLOCK,
            )

        self._temporary_human_blocker = TemporaryDrawerArmBlocker(
            model=self._mojo.physics.model.ptr,
            data=self._mojo.physics.data.ptr,
            human_joint_names=human_joint_names,
            ee_site_name=config["ee_site_name"],
            handle_site_name=config["handle_site_name"],
            q_outside=q_outside,
            q_block=q_block,
            q_exit=config["q_exit"],
            trigger_dist=config["trigger_dist"],
            enter_duration=config["enter_duration"],
            hold_duration=config["hold_duration"],
            exit_duration=config["exit_duration"],
            natural_motion_scale=config["natural_motion_scale"],
            exit_after_y_peak=config["exit_after_y_peak"],
            max_joint_speed=config["max_blocker_joint_speed"],
        )
        self._temporary_human_blocker_info = dict(self._temporary_human_blocker.last_info)

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

    def _get_task_privileged_obs_space(self) -> dict[str, Any]:
        num_human_qpos = sum(len(human.get_state()["qpos"]) for human in self.humanarms)
        num_human_qvel = sum(len(human.get_state()["qvel"]) for human in self.humanarms)
        return {
            "human_arm_qpos": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_human_qpos,),
                dtype=np.float32,
            ),
            "human_arm_qvel": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_human_qvel,),
                dtype=np.float32,
            ),
        }

    def _get_task_info(self) -> dict[str, Any]:
        info = dict(self._temporary_human_blocker_info)
        drawer_open_distance = self._drawer_open_distance()
        if drawer_open_distance is not None:
            info["drawer_open_distance"] = drawer_open_distance
        return info

    def _drawer_open_distance(self):
        if not hasattr(self, "_mojo"):
            return None
        model = self._mojo.physics.model.ptr
        data = self._mojo.physics.data.ptr
        for name in ("base_cabinet_600/drawer_small_4", "drawer_small_4"):
            try:
                joint_id = _resolve_mujoco_name(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
                return float(data.qpos[int(model.jnt_qposadr[joint_id])])
            except ValueError:
                continue
        if hasattr(self, "cabinet_drawers"):
            return float(self.cabinet_drawers.get_state()[-1])
        return None

    def _on_step(self, arm_action:np.ndarray=None):
        if self._temporary_human_blocker is not None:
            self._temporary_human_blocker_info = self._temporary_human_blocker.update(self.get_dt())
            return

        human: HumanArm
        for human in self.humanarms:
            if arm_action is not None:
                human.set_qpos_target(arm_action)
            human._on_step(self.get_dt())

    def _on_reset(self, seed: Optional[int] = None):
        for human in self.humanarms:
            human.reset(seed=seed)
        if self._temporary_human_blocker is not None:
            self._temporary_human_blocker_info = self._temporary_human_blocker.reset()

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
        if self._temporary_human_blocker is not None:
            self._temporary_human_blocker_info = self._temporary_human_blocker.reset()
        return self.get_observation(), self.get_info()



class HumanArmDrawerTopOpen(_HumanArmCupboardsInteractionEnv):
    """Open top drawer of the cupboard task."""

    def _success(self) -> bool:
        if not np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE):
            return False
        # for human in self.humanarms:
        #     if not human.is_colliding(self.cabinet_wall.shelf_bottom):
        #         return False
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
        # for human in self.humanarms:
        #     if not human.is_colliding(self.cabinet_wall.shelf_bottom):
        #         return False
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
