from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import mujoco

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import KinematicProp
from copy import deepcopy

ControlMode = Literal["scripted", "position", "torque"]


@dataclass
class PDGains:
    kp: np.ndarray
    kd: np.ndarray

@dataclass
class PrimitivePlan:
    name: str
    q_points: list[np.ndarray]   # [q0, q1, q2, ...]
    durations: list[float]       # per segment duration
    hold: float = 0.0

class HumanArm(KinematicProp):
    """
    Human-like kinematic cylinder arm with:
      - style-conditioned motion primitives
      - minimum-jerk joint transitions
      - goal-driven carrier drift in XY
      - geometric keepout against robot geoms
      - no carrier rollback / no mid-episode phase reset

    Notes:
      - get_state() is intentionally kept unchanged.
      - The arm remains kinematic in the main env.
      - Keepout only modifies carrier XY motion, not joint progression.
    """

    _ARM_XML: Path = ASSETS_PATH / "props/human_arm/arm_two_joints.xml"
    _ROOT_PREFIX: str = "cylinder_arm"

    _JOINT_NAMES: Tuple[str, str, str, str] = (
        "arm_shoulder_base",
        "arm_shoulder_yaw",
        "arm_shoulder_pitch",
        "arm_elbow",
    )
    _ACT_NAMES: Tuple[str, str, str, str] = (
        "act_arm_shoulder_base",
        "act_arm_shoulder_yaw",
        "act_arm_shoulder_pitch",
        "act_arm_elbow",
    )
    _CARRIER_JOINTS: Tuple[str, str] = ("arm_tx", "arm_ty")

    _GAINS: PDGains = PDGains(
        kp=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        kd=np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float64),
    )

    # Keepout
    MIN_CLEAR: float = 0.010
    KEEP_SOFT: float = 0.013
    KEEP_HARD: float = 0.003
    KEEPOUT_SCALE: float = 0.75
    MAX_KEEP_PUSH: float = 0.015  # max outward bias per step

    # Carrier workspace
    _walk_enable: bool = True
    _walk_radius: float = 0.25

    # Motion timing
    _CURRENT_TIME: float = 0.0
    _MOTION_FPS: float = 30.0
    _PRIM_DURATION_RANGE: Tuple[float, float] = (2.5, 5.5)
    _DWELL_RANGE: Tuple[float, float] = (0.4, 1.1)

    # Primitive names
    _PRIMITIVES: Tuple[str, ...] = (
        "idle",
        "sweep_lr",
        "sweep_rl",
        "reach",
        "retract",
        "lift_lower",
    )
    TWIST_SCALE: float = 0.4

    def __init__(self, mojo, kinematic=None, cache_colliders=None, cache_sites=None, parent=None, **kwargs):
        super().__init__(mojo, kinematic, cache_colliders, cache_sites, parent, **kwargs)

        self._physics = self._mojo.physics
        m = self._physics.model.ptr
        d = self._physics.data

        # IDs / addresses
        self._joint_ids = np.array([self._get_joint_id(n) for n in self._JOINT_NAMES], dtype=np.int32)
        self._act_id = np.array([self._get_actuator_id(n) for n in self._ACT_NAMES], dtype=np.int32)
        self._qpos_adr = np.array([self._get_joint_qpos_adr(n) for n in self._JOINT_NAMES], dtype=np.int32)
        self._qvel_adr = np.array([self._get_joint_dof_adr(n) for n in self._JOINT_NAMES], dtype=np.int32)
        self._carrier_qpos_adr = np.array([self._get_joint_qpos_adr(n) for n in self._CARRIER_JOINTS], dtype=np.int32)
        self._carrier_qvel_adr = np.array([self._get_joint_dof_adr(n) for n in self._CARRIER_JOINTS], dtype=np.int32)

        # Joint limits
        self._joint_range = np.asarray(m.jnt_range[self._joint_ids], dtype=np.float64)  # (4,2)

        # Ctrl clip
        self._CTRL_CLIP = 1.0
        try:
            cr = self._physics.model.actuator_ctrlrange[self._act_id, :]
            self._CTRL_CLIP = float(np.max(np.abs(cr)))
        except Exception:
            pass

        # RNG
        self._rng = np.random.default_rng()

        # External control state
        self._mode: ControlMode = "scripted"
        self._qpos_target = np.zeros(len(self._JOINT_NAMES), dtype=np.float64)
        self._ctrl_target = np.zeros(len(self._ACT_NAMES), dtype=np.float64)

        # Filtered kinematic state (2nd-order tracking)
        self._qpos_filt = d.qpos[self._qpos_adr].copy()
        self._qvel_filt = np.zeros(len(self._JOINT_NAMES), dtype=np.float64)

        # Carrier state
        self._walk_center_xy = d.qpos[self._carrier_qpos_adr].copy()
        self._walk_xy = np.zeros(2, dtype=np.float64)
        self._walk_v = np.zeros(2, dtype=np.float64)
        self._walk_goal_xy = np.zeros(2, dtype=np.float64)
        self._carrier_dwell = 0.0
        self._soft_keepout_time = 0.0
        self._last_goal_refresh_t = 0.0
        
        # Style latent (sampled each reset)
        self._style_speed = 1.0
        self._style_amp = 1.0
        self._style_elbow_bias = 0.0
        self._style_pitch_bias = 0.0
        self._style_center_bias = np.zeros(2, dtype=np.float64)
        self._style_dwell = 1.0

        # Primitive state
        self._primitive_plan: Optional[PrimitivePlan] = None
        self._primitive_seg_idx: int = 0
        self._primitive_seg_t: float = 0.0
        self._primitive_hold_t: float = 0.0
        self._last_primitive_name: Optional[str] = None

        # Arm geoms
        self._arm_geom_ids = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, self._pref("upperarm_geom")),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, self._pref("forearm_geom")),
        ], dtype=np.int32)
        if np.any(self._arm_geom_ids < 0):
            raise RuntimeError(f"Arm geom ids not found: {self._arm_geom_ids} (check name prefixing)")

        # Robot keepout geoms
        robot_ids = []
        for gid in range(int(m.ngeom)):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not name.startswith("h1/"):
                continue
            if int(m.geom_contype[gid]) == 0 or int(m.geom_conaffinity[gid]) == 0:
                continue
            if float(m.geom_rbound[gid]) <= 0.02:
                continue
            robot_ids.append(gid)

        self._robot_geom_ids = np.array(robot_ids, dtype=np.int32)
        self._robot_keepout_r = self.KEEPOUT_SCALE * np.asarray(
            m.geom_rbound[self._robot_geom_ids], dtype=np.float64
        )

        self.dt_frame = 1.0 / max(float(self._MOTION_FPS), 1e-9)
        self._keepout_nxy_filt = np.zeros(2, dtype=np.float64)
        
        # Debug
        self._debug_keepout_clear = np.inf
        self._debug_keepout_active = False
        self._debug_keepout_nxy = np.zeros(2, dtype=np.float64)
        self._debug_keepout_push = 0.0
        self._debug_keepout_zone = "free"

    # --------------------
    # Required by Prop API
    # --------------------
    @property
    def _model_path(self) -> Path:
        return self._ARM_XML

    # --------------------
    # Name helpers
    # --------------------
    def get_debug_keepout_state(self):
        return {
            "clear": float(self._debug_keepout_clear),
            "active": bool(self._debug_keepout_active),
            "nxy": self._debug_keepout_nxy.copy(),
            "push": float(self._debug_keepout_push),
            "zone": self._debug_keepout_zone,
        }

    def _pref(self, name: str) -> str:
        return f"{self._ROOT_PREFIX}/{name}"

    def _get_joint_id(self, joint_name: str) -> int:
        full = self._pref(joint_name)
        try:
            return self._physics.model.joint(full).id
        except Exception as e:
            raise RuntimeError(f"Cannot find joint '{full}' in physics.model") from e

    def _get_actuator_id(self, act_name: str) -> int:
        full = self._pref(act_name)
        try:
            return self._physics.model.actuator(full).id
        except Exception as e:
            raise RuntimeError(f"Cannot find actuator '{full}' in physics.model") from e

    def _get_joint_qpos_adr(self, joint_name: str) -> int:
        full = self._pref(joint_name)
        try:
            jid = self._physics.model.joint(full).id
            return int(self._physics.model.ptr.jnt_qposadr[jid])
        except Exception as e:
            raise RuntimeError(f"Cannot find joint '{full}' (qpos adr) in physics.model") from e

    def _get_joint_dof_adr(self, joint_name: str) -> int:
        full = self._pref(joint_name)
        try:
            jid = self._physics.model.joint(full).id
            return int(self._physics.model.ptr.jnt_dofadr[jid])
        except Exception as e:
            raise RuntimeError(f"Cannot find joint '{full}' (dof adr) in physics.model") from e

    # --------------------
    # External control API
    # --------------------
    def export_internal_state(self):
        return {
            "mode": self._mode,
            "qpos_target": self._qpos_target.copy(),
            "ctrl_target": self._ctrl_target.copy(),
            "qpos_filt": self._qpos_filt.copy(),
            "qvel_filt": self._qvel_filt.copy(),
            "walk_center_xy": self._walk_center_xy.copy(),
            "walk_xy": self._walk_xy.copy(),
            "walk_v": self._walk_v.copy(),
            "walk_goal_xy": self._walk_goal_xy.copy(),
            "carrier_dwell": float(self._carrier_dwell),
            "style_speed": float(self._style_speed),
            "style_amp": float(self._style_amp),
            "style_elbow_bias": float(self._style_elbow_bias),
            "style_pitch_bias": float(self._style_pitch_bias),
            "style_center_bias": self._style_center_bias.copy(),
            "style_dwell": float(self._style_dwell),
            "primitive_plan": deepcopy(self._primitive_plan),
            "primitive_seg_idx": int(self._primitive_seg_idx),
            "primitive_seg_t": float(self._primitive_seg_t),
            "primitive_hold_t": float(self._primitive_hold_t),
            "last_primitive_name": self._last_primitive_name,
            "current_time": float(self._CURRENT_TIME),
            "rng_state": deepcopy(self._rng.bit_generator.state),
            "keepout_nxy_filt": self._keepout_nxy_filt.copy(),
        }
    
    def import_internal_state(self, state):
        self._mode = state["mode"]
        self._qpos_target[:] = state["qpos_target"]
        self._ctrl_target[:] = state["ctrl_target"]
        self._qpos_filt = state["qpos_filt"].copy()
        self._qvel_filt = state["qvel_filt"].copy()
        self._walk_center_xy = state["walk_center_xy"].copy()
        self._walk_xy = state["walk_xy"].copy()
        self._walk_v = state["walk_v"].copy()
        self._walk_goal_xy = state["walk_goal_xy"].copy()
        self._carrier_dwell = float(state["carrier_dwell"])
        self._style_speed = float(state["style_speed"])
        self._style_amp = float(state["style_amp"])
        self._style_elbow_bias = float(state["style_elbow_bias"])
        self._style_pitch_bias = float(state["style_pitch_bias"])
        self._style_center_bias = state["style_center_bias"].copy()
        self._style_dwell = float(state["style_dwell"])
        self._primitive_plan = deepcopy(state["primitive_plan"])
        self._primitive_seg_idx = int(state["primitive_seg_idx"])
        self._primitive_seg_t = float(state["primitive_seg_t"])
        self._primitive_hold_t = float(state["primitive_hold_t"])
        self._last_primitive_name = state["last_primitive_name"]
        self._CURRENT_TIME = float(state["current_time"])
        if getattr(self, "_rng", None) is None:
            self._rng = np.random.default_rng()
        self._rng.bit_generator.state = deepcopy(state["rng_state"])
        self._keepout_nxy_filt = state["keepout_nxy_filt"].copy()

    def set_mode(self, mode: ControlMode):
        if mode not in ("scripted", "position", "torque"):
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode

    def set_qpos_target(self, q_des: np.ndarray, *, radians: bool = True):
        q = np.asarray(q_des, dtype=np.float64).reshape(len(self._JOINT_NAMES),)
        if not radians:
            q = np.deg2rad(q)
        self._qpos_target[:] = q
        self._mode = "position"

    def set_ctrl(self, u: np.ndarray):
        u = np.asarray(u, dtype=np.float64).reshape(len(self._ACT_NAMES),)
        self._ctrl_target[:] = u
        self._mode = "torque"

    def get_state(self):
        """Convenience accessor for debugging/obs (used externally)."""
        d = self._physics.data
        q = d.qpos[self._qpos_adr].copy()
        qd = d.qvel[self._qvel_adr].copy()
        return {"qpos": q, "qvel": qd, "mode": self._mode}

    # --------------------
    # Utility
    # --------------------
    def _clip_joint_vec(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).copy()
        lo = self._joint_range[:, 0]
        hi = self._joint_range[:, 1]
        return np.clip(q, lo, hi)

    def _minimum_jerk(self, tau: float) -> float:
        tau = float(np.clip(tau, 0.0, 1.0))
        return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    def _rand_disk(self, radius: float) -> np.ndarray:
        r = radius * np.sqrt(self._rng.uniform())
        th = self._rng.uniform(0.0, 2.0 * np.pi)
        return np.array([r * np.cos(th), r * np.sin(th)], dtype=np.float64)

    # --------------------
    # Style / primitives
    # --------------------
    def _sample_style(self):
        self._style_speed = self._rng.uniform(0.85, 1.25)
        self._style_amp = self._rng.uniform(0.85, 1.20)
        self._style_elbow_bias = np.deg2rad(self._rng.uniform(-10.0, 12.0))
        self._style_pitch_bias = np.deg2rad(self._rng.uniform(-8.0, 8.0))
        self._style_center_bias = self._rand_disk(0.06)
        self._style_dwell = self._rng.uniform(0.8, 1.3)

    def _neutral_pose(self) -> np.ndarray:
        q = np.array([
            np.deg2rad(self._rng.uniform(-8.0, 8.0)),   # base
            np.deg2rad(self._rng.uniform(-10.0, 10.0)),   # yaw
            np.deg2rad(-20.0) + self._style_pitch_bias,   # pitch
            np.deg2rad(65.0) + self._style_elbow_bias,    # elbow
        ], dtype=np.float64)
        return self._clip_joint_vec(q)
    
    def _task_anchor_pose(self) -> np.ndarray:
        q = np.array([
            np.deg2rad(self._rng.uniform(-10.0, 10.0)),   # base
            np.deg2rad(self._rng.uniform(-12.0, 12.0)),   # yaw
            np.deg2rad(self._rng.uniform(-35.0, -10.0)) + self._style_pitch_bias,
            np.deg2rad(self._rng.uniform(50.0, 85.0)) + self._style_elbow_bias,
        ], dtype=np.float64)
        return self._clip_joint_vec(q)

    def _sample_primitive_name(self) -> str:
        candidates = ["reach", "retract", "sweep_lr", "sweep_rl", "lift_lower", "idle"]
        probs = np.array([0.26, 0.18, 0.22, 0.22, 0.09, 0.03], dtype=np.float64)

        if self._last_primitive_name is not None and self._last_primitive_name in candidates:
            probs[candidates.index(self._last_primitive_name)] *= 0.12

        # prefer alternating sweep direction
        if self._last_primitive_name == "sweep_lr":
            probs[candidates.index("sweep_rl")] *= 1.6
        elif self._last_primitive_name == "sweep_rl":
            probs[candidates.index("sweep_lr")] *= 1.6

        probs /= probs.sum()
        return str(self._rng.choice(candidates, p=probs))

    def _build_primitive_plan(self, name: str, q_start: np.ndarray) -> PrimitivePlan:
        amp = self._style_amp
        tw = self.TWIST_SCALE
        q_neutral = self._task_anchor_pose()

        if name == "idle":
            q_mid = q_neutral + np.array([
                tw * np.deg2rad(self._rng.uniform(-6.0, 6.0)),
                tw * np.deg2rad(self._rng.uniform(-8.0, 8.0)),
                np.deg2rad(self._rng.uniform(-6.0, 4.0)),
                np.deg2rad(self._rng.uniform(-6.0, 6.0)),
            ], dtype=np.float64)
            q_end = q_neutral + np.array([
                tw * np.deg2rad(self._rng.uniform(-4.0, 4.0)),
                tw * np.deg2rad(self._rng.uniform(-5.0, 5.0)),
                np.deg2rad(self._rng.uniform(-4.0, 4.0)),
                np.deg2rad(self._rng.uniform(-4.0, 4.0)),
            ], dtype=np.float64)

            return PrimitivePlan(
                name=name,
                q_points=[self._clip_joint_vec(q_start), self._clip_joint_vec(q_mid), self._clip_joint_vec(q_end)],
                durations=[1.8, 2.2],
                hold=self._rng.uniform(0.6, 1.0),
            )

        elif name == "reach":
            q_prep = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(-4.0, 4.0)),
                tw * np.deg2rad(self._rng.uniform(-6.0, 6.0)),
                np.deg2rad(self._rng.uniform(-15.0, -6.0)),
                np.deg2rad(self._rng.uniform(5.0, 12.0)),
            ], dtype=np.float64)

            q_reach = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(-8.0, 8.0)),
                tw * np.deg2rad(self._rng.uniform(-10.0, 10.0)),
                np.deg2rad(self._rng.uniform(-48.0, -28.0)),
                np.deg2rad(self._rng.uniform(-28.0, -8.0)),
            ], dtype=np.float64)

            q_settle = q_reach + np.array([
                tw * np.deg2rad(self._rng.uniform(-3.0, 3.0)),
                tw * np.deg2rad(self._rng.uniform(-3.0, 3.0)),
                np.deg2rad(self._rng.uniform(2.0, 6.0)),
                np.deg2rad(self._rng.uniform(2.0, 8.0)),
            ], dtype=np.float64)

            return PrimitivePlan(
                name=name,
                q_points=[
                    self._clip_joint_vec(q_start),
                    self._clip_joint_vec(q_prep),
                    self._clip_joint_vec(q_reach),
                    self._clip_joint_vec(q_settle),
                ],
                durations=[1.6, 2.2, 1.4],
                hold=self._rng.uniform(0.6, 1.2),
            )

        elif name == "retract":
            q_mid = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(-6.0, 6.0)),
                tw * np.deg2rad(self._rng.uniform(-10.0, 10.0)),
                np.deg2rad(self._rng.uniform(2.0, 10.0)),
                np.deg2rad(self._rng.uniform(14.0, 28.0)),
            ], dtype=np.float64)

            q_end = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(-5.0, 5.0)),
                tw * np.deg2rad(self._rng.uniform(-8.0, 8.0)),
                np.deg2rad(self._rng.uniform(6.0, 14.0)),
                np.deg2rad(self._rng.uniform(18.0, 34.0)),
            ], dtype=np.float64)

            return PrimitivePlan(
                name=name,
                q_points=[self._clip_joint_vec(q_start), self._clip_joint_vec(q_mid), self._clip_joint_vec(q_end)],
                durations=[1.6, 1.8],
                hold=self._rng.uniform(0.4, 0.8),
            )

        elif name == "sweep_lr":
            q_prep = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(2.0, 6.0)),
                tw * np.deg2rad(self._rng.uniform(-4.0, 4.0)),
                np.deg2rad(self._rng.uniform(-16.0, -8.0)),
                np.deg2rad(self._rng.uniform(2.0, 10.0)),
            ], dtype=np.float64)

            q_cross = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(6.0, 14.0)),
                tw * np.deg2rad(self._rng.uniform(8.0, 16.0)),
                np.deg2rad(self._rng.uniform(-28.0, -14.0)),
                np.deg2rad(self._rng.uniform(-10.0, 10.0)),
            ], dtype=np.float64)

            q_exit = q_cross + np.array([
                tw * np.deg2rad(self._rng.uniform(-4.0, 2.0)),
                tw * np.deg2rad(self._rng.uniform(-5.0, 3.0)),
                np.deg2rad(self._rng.uniform(2.0, 6.0)),
                np.deg2rad(self._rng.uniform(2.0, 6.0)),
            ], dtype=np.float64)

            return PrimitivePlan(
                name=name,
                q_points=[
                    self._clip_joint_vec(q_start),
                    self._clip_joint_vec(q_prep),
                    self._clip_joint_vec(q_cross),
                    self._clip_joint_vec(q_exit),
                ],
                durations=[1.3, 2.4, 1.4],
                hold=self._rng.uniform(0.4, 0.9),
            )

        elif name == "sweep_rl":
            q_prep = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(-6.0, -2.0)),
                tw * np.deg2rad(self._rng.uniform(-4.0, 4.0)),
                np.deg2rad(self._rng.uniform(-16.0, -8.0)),
                np.deg2rad(self._rng.uniform(2.0, 10.0)),
            ], dtype=np.float64)

            q_cross = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(-14.0, -6.0)),
                tw * np.deg2rad(self._rng.uniform(-16.0, -8.0)),
                np.deg2rad(self._rng.uniform(-28.0, -14.0)),
                np.deg2rad(self._rng.uniform(-10.0, 10.0)),
            ], dtype=np.float64)

            q_exit = q_cross + np.array([
                tw * np.deg2rad(self._rng.uniform(-2.0, 4.0)),
                tw * np.deg2rad(self._rng.uniform(-3.0, 5.0)),
                np.deg2rad(self._rng.uniform(2.0, 6.0)),
                np.deg2rad(self._rng.uniform(2.0, 6.0)),
            ], dtype=np.float64)

            return PrimitivePlan(
                name=name,
                q_points=[
                    self._clip_joint_vec(q_start),
                    self._clip_joint_vec(q_prep),
                    self._clip_joint_vec(q_cross),
                    self._clip_joint_vec(q_exit),
                ],
                durations=[1.3, 2.4, 1.4],
                hold=self._rng.uniform(0.4, 0.9),
            )

        elif name == "lift_lower":
            q_up = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(-8.0, 8.0)),
                tw * np.deg2rad(self._rng.uniform(-8.0, 8.0)),
                np.deg2rad(self._rng.uniform(-34.0, -18.0)),
                np.deg2rad(self._rng.uniform(-2.0, 10.0)),
            ], dtype=np.float64)

            q_down = q_neutral + amp * np.array([
                tw * np.deg2rad(self._rng.uniform(-6.0, 6.0)),
                tw * np.deg2rad(self._rng.uniform(-6.0, 6.0)),
                np.deg2rad(self._rng.uniform(-12.0, -4.0)),
                np.deg2rad(self._rng.uniform(8.0, 20.0)),
            ], dtype=np.float64)

            return PrimitivePlan(
                name=name,
                q_points=[self._clip_joint_vec(q_start), self._clip_joint_vec(q_up), self._clip_joint_vec(q_down)],
                durations=[1.8, 2.0],
                hold=self._rng.uniform(0.4, 0.9),
            )

        else:
            return PrimitivePlan(
                name="idle",
                q_points=[self._clip_joint_vec(q_start), self._task_anchor_pose()],
                durations=[1.5],
                hold=0.3,
            )
        
    def _start_new_primitive(self, q_start: np.ndarray):
        name = self._sample_primitive_name()
        plan = self._build_primitive_plan(name, q_start)

        # style-based time scaling
        plan.durations = [float(np.clip(d / self._style_speed, 1.0, 4.5)) for d in plan.durations]
        plan.hold = float(np.clip(plan.hold / self._style_speed, 0.2, 1.5))

        self._primitive_plan = plan
        self._primitive_seg_idx = 0
        self._primitive_seg_t = 0.0
        self._primitive_hold_t = 0.0
        self._last_primitive_name = name

    def _scripted_q_target(self, dt: float) -> np.ndarray:
        if self._primitive_plan is None:
            self._start_new_primitive(self._qpos_filt.copy())

        plan = self._primitive_plan

        # hold phase at end
        if self._primitive_seg_idx >= len(plan.durations):
            self._primitive_hold_t += dt
            q = plan.q_points[-1].copy()

            if self._primitive_hold_t >= plan.hold:
                self._start_new_primitive(q.copy())
                plan = self._primitive_plan
                q = plan.q_points[0].copy()

            return self._clip_joint_vec(q)

        self._primitive_seg_t += dt

        q0 = plan.q_points[self._primitive_seg_idx]
        q1 = plan.q_points[self._primitive_seg_idx + 1]
        T = max(plan.durations[self._primitive_seg_idx], 1e-6)

        tau = self._primitive_seg_t / T
        s = self._minimum_jerk(tau)
        q = q0 + s * (q1 - q0)

        if self._primitive_seg_t >= T:
            self._primitive_seg_idx += 1
            self._primitive_seg_t = 0.0

        return self._clip_joint_vec(q)

    # --------------------
    # Carrier motion
    # --------------------
    def _sample_new_goal(self):
        local = self._style_center_bias + self._rand_disk(0.8 * self._walk_radius)
        r = float(np.linalg.norm(local))
        if r > self._walk_radius:
            local = local / (r + 1e-12) * self._walk_radius
        self._walk_goal_xy = local

    def _advance_carrier_nominal(self, dt: float) -> np.ndarray:
        if not self._walk_enable:
            return np.zeros(2, dtype=np.float64)

        if self._carrier_dwell > 0.0:
            self._carrier_dwell = max(0.0, self._carrier_dwell - dt)
            desired_v = np.zeros(2, dtype=np.float64)
        else:
            err = self._walk_goal_xy - self._walk_xy
            dist = float(np.linalg.norm(err))

            if dist < 0.025:
                self._carrier_dwell = self._rng.uniform(*self._DWELL_RANGE) * self._style_dwell
                self._sample_new_goal()
                err = self._walk_goal_xy - self._walk_xy
                dist = float(np.linalg.norm(err))

            vmax = 0.16 * self._style_speed
            kp = 1.6 * self._style_speed
            desired_v = kp * err
            dnorm = float(np.linalg.norm(desired_v))
            if dnorm > vmax:
                desired_v = desired_v / (dnorm + 1e-12) * vmax

        # smooth 2nd-order-like tracking of desired velocity
        tau_v = 0.35
        alpha_v = 1.0 - np.exp(-dt / tau_v)
        self._walk_v = (1.0 - alpha_v) * self._walk_v + alpha_v * desired_v

        # very small low-frequency perturbation to avoid being too learnable
        self._walk_v += 0.004 * np.sqrt(max(dt, 1e-9)) * self._rng.standard_normal(2)

        # Slow carrier down when already near keepout boundary.
        # This prevents repeatedly entering hard zone and causing visible stutter.
        clear = float(getattr(self, "_debug_keepout_clear", np.inf))

        base_vmax = 0.18 * self._style_speed
        vmax = base_vmax

        if clear < self.KEEP_SOFT:
            alpha = np.clip(
                (clear - self.KEEP_HARD) / (self.KEEP_SOFT - self.KEEP_HARD + 1e-12),
                0.0,
                1.0,
            )
            # near hard zone -> slower, but not near-zero
            vmax = base_vmax * (0.55 + 0.45 * alpha)

        vnorm = float(np.linalg.norm(self._walk_v))
        if vnorm > vmax:
            self._walk_v = self._walk_v / (vnorm + 1e-12) * vmax

        dx = self._walk_v * dt
        return dx

    # --------------------
    # Keepout geometry
    # --------------------
    def _set_kinematic_state(self, carrier_xy: np.ndarray, joint_q: np.ndarray):
        self._physics.data.qpos[self._carrier_qpos_adr] = carrier_xy
        self._physics.data.qvel[self._carrier_qvel_adr] = 0.0

        self._physics.data.qpos[self._qpos_adr] = joint_q
        self._physics.data.qvel[self._qvel_adr] = 0.0
        self._physics.data.ctrl[self._act_id] = 0.0

        self._physics.forward()

    def _min_keepout_clearance(self) -> Tuple[float, Optional[np.ndarray]]:
        m = self._physics.model.ptr
        d = self._physics.data.ptr
        keep_ids = self._robot_geom_ids

        if keep_ids is None or len(keep_ids) == 0:
            return float("inf"), None

        X = np.asarray(d.geom_xpos[keep_ids], dtype=np.float64)   # (N,3)
        Rb = self._robot_keepout_r

        best_clear = float("inf")
        best_nxy = None

        for gid in map(int, self._arm_geom_ids):
            size = np.asarray(m.geom_size[gid], dtype=np.float64)
            r_arm = float(size[0])
            half = float(size[1])

            xg = np.asarray(d.geom_xpos[gid], dtype=np.float64)
            Rg = np.asarray(d.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            z = Rg[:, 2]

            a = xg - half * z
            b = xg + half * z
            ab = b - a
            ab2 = float(np.dot(ab, ab)) + 1e-12

            Xa = X - a[None, :]
            tseg = (Xa @ ab) / ab2
            tseg = np.clip(tseg, 0.0, 1.0)
            Q = a[None, :] + tseg[:, None] * ab[None, :]

            V = X - Q
            dist_center = np.linalg.norm(V, axis=1)
            clear = dist_center - (r_arm + Rb)

            j = int(np.argmin(clear))
            cj = float(clear[j])
            if cj < best_clear:
                best_clear = cj
                n = -V[j] / (float(dist_center[j]) + 1e-12)   # outward normal in world
                best_nxy = n[:2].copy()

        return best_clear, best_nxy

    def _solve_carrier_xy(self, prev_xy: np.ndarray, dx_nominal: np.ndarray, joint_q: np.ndarray, dt: float) -> np.ndarray:
        """
        Soft/hard/emergency keepout:
        - free zone: no correction
        - soft zone: remove inward component only
        - hard zone: strong outward-biased correction
        - emergency zone: clear is deeply negative, force push out
        """
        candidate = prev_xy + dx_nominal

        # keep inside walk disk
        rel = candidate - self._walk_center_xy
        rad = float(np.linalg.norm(rel))
        if rad > self._walk_radius:
            rel = rel / (rad + 1e-12) * self._walk_radius
            candidate = self._walk_center_xy + rel

        self._set_kinematic_state(candidate, joint_q)
        best_clear, nxy = self._min_keepout_clearance()

        # debug defaults
        self._debug_keepout_clear = float(best_clear)
        self._debug_keepout_active = bool(best_clear < self.MIN_CLEAR)
        self._debug_keepout_push = 0.0
        self._debug_keepout_zone = "free"
        self._debug_keepout_nxy = np.zeros(2, dtype=np.float64)

        if nxy is not None:
            beta_n = 1.0 - np.exp(-dt / 0.25)
            self._keepout_nxy_filt = (1.0 - beta_n) * self._keepout_nxy_filt + beta_n * nxy
            nrm_f = float(np.linalg.norm(self._keepout_nxy_filt))
            if nrm_f > 1e-9:
                nxy = self._keepout_nxy_filt / nrm_f
                self._debug_keepout_nxy = nxy.copy()

        # ---------- free zone ----------
        if nxy is None or best_clear >= self.KEEP_SOFT:
            self._soft_keepout_time = 0.0
            return candidate

        dx = candidate - prev_xy
        self._debug_keepout_active = True

        # ---------- soft zone ----------
        if best_clear >= self.KEEP_HARD:
            self._debug_keepout_zone = "soft"

            nrm = float(np.linalg.norm(nxy))
            if nrm > 1e-9:
                nxy = nxy / nrm
                self._debug_keepout_nxy = nxy.copy()

            # If we stay in soft zone for a while, refresh the wander goal
            # so the arm does not keep rubbing along the same boundary.
            self._soft_keepout_time += dt
            if self._soft_keepout_time > 0.35:
                self._sample_new_goal()
                self._carrier_dwell = max(self._carrier_dwell, 0.08)
                self._soft_keepout_time = 0.0

            return candidate
        
        # ---------- emergency zone ----------
        if best_clear < -0.02:
            self._soft_keepout_time = 0.0
            self._debug_keepout_zone = "emergency"

            nrm = float(np.linalg.norm(nxy))
            if nrm < 1e-9:
                return prev_xy.copy()

            nxy = nxy / nrm
            self._debug_keepout_nxy = nxy.copy()

            # stop carrier aggressively
            self._walk_v[:] = 0.0
            self._carrier_dwell = max(self._carrier_dwell, 0.35)

            dx = 0.015 * nxy
            self._debug_keepout_push = 0.015

            candidate = prev_xy + dx

            rel = candidate - self._walk_center_xy
            rad = float(np.linalg.norm(rel))
            if rad > self._walk_radius:
                rel = rel / (rad + 1e-12) * self._walk_radius
                candidate = self._walk_center_xy + rel

            self._set_kinematic_state(candidate, joint_q)
            return candidate

        # ---------- hard zone ----------
        self._debug_keepout_zone = "hard"
        self._soft_keepout_time = 0.0
        nrm = float(np.linalg.norm(nxy))
        if nrm < 1e-9:
            return prev_xy.copy()

        nxy = nxy / nrm
        self._debug_keepout_nxy = nxy.copy()

        # remove inward carrier velocity too, not just current-step displacement
        vn_v = float(np.dot(self._walk_v, nxy))
        if vn_v < 0.0:
            self._walk_v = self._walk_v - vn_v * nxy
        self._walk_v *= 0.75

        dn = float(np.dot(dx, nxy))
        if dn < 0.0:
            dx = dx - dn * nxy

        tangent = dx - np.dot(dx, nxy) * nxy
        corr = float(np.clip(self.MIN_CLEAR - best_clear, 0.0, self.MAX_KEEP_PUSH))
        self._debug_keepout_push = corr

        # strong outward bias, weak tangential motion
        dx = 0.35 * tangent + max(0.004, 0.6 * corr) * nxy

        # dwell a bit so the goal tracker doesn't immediately push back in
        self._carrier_dwell = max(self._carrier_dwell, 0.06)

        candidate = prev_xy + dx

        rel = candidate - self._walk_center_xy
        rad = float(np.linalg.norm(rel))
        if rad > self._walk_radius:
            rel = rel / (rad + 1e-12) * self._walk_radius
            candidate = self._walk_center_xy + rel

        self._set_kinematic_state(candidate, joint_q)
        return candidate
    # --------------------
    # Episode lifecycle
    # --------------------
    def reset(self, seed=None, time: float = 0.0, qpos0: Optional[np.ndarray] = None, mode: Optional[ControlMode] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._CURRENT_TIME = float(time)
        if mode is not None:
            self._mode = mode

        self._sample_style()

        if qpos0 is None:
            qpos0 = self._neutral_pose()
        else:
            qpos0 = np.asarray(qpos0, dtype=np.float64).reshape(4,)

        d = self._physics.data
        d.qpos[self._qpos_adr] = qpos0
        d.qvel[self._qvel_adr] = 0.0
        d.ctrl[self._act_id] = 0.0

        self._qpos_target[:] = qpos0
        self._ctrl_target[:] = 0.0
        self._qpos_filt = qpos0.copy()
        self._qvel_filt = np.zeros_like(self._qpos_filt)

        self._walk_center_xy = np.zeros(2, dtype=np.float64)
        d.qpos[self._carrier_qpos_adr] = self._walk_center_xy
        d.qvel[self._carrier_qvel_adr] = 0.0

        self._walk_xy = self._walk_center_xy.copy()
        self._walk_v = np.zeros(2, dtype=np.float64)
        self._sample_new_goal()
        self._carrier_dwell = self._rng.uniform(*self._DWELL_RANGE) * self._style_dwell
        self._soft_keepout_time = 0.0
        self._last_goal_refresh_t = 0.0

        self._primitive_plan = None
        self._primitive_seg_idx = 0
        self._primitive_seg_t = 0.0
        self._primitive_hold_t = 0.0
        self._last_primitive_name = None
        self._start_new_primitive(qpos0.copy())

        self._physics.forward()

    # --------------------
    # Core step
    # --------------------
    def _on_step(self, dt: float):
        self._CURRENT_TIME += float(dt)
        dt = float(dt)
        phys = self._physics

        # Torque mode unchanged
        if self._mode == "torque":
            u = self._ctrl_target
            clip = float(self._CTRL_CLIP)
            if clip > 0:
                u = np.clip(u, -clip, clip)
            phys.data.ctrl[self._act_id] = u
            phys.forward()
            return

        # Joint target
        if self._mode == "scripted":
            q_des = self._scripted_q_target(dt)
            self._qpos_target[:] = q_des
        elif self._mode == "position":
            q_des = self._clip_joint_vec(self._qpos_target)
        else:
            raise RuntimeError(f"Unknown control mode: {self._mode}")

        # 2nd-order critically damped tracking for smoother motion
        omega = 3.0
        zeta = 1.2
        acc = omega * omega * (q_des - self._qpos_filt) - 2.0 * zeta * omega * self._qvel_filt

        # acceleration limits (rad/s^2)
        acc_max = np.deg2rad(np.array([90.0, 90.0, 110.0, 140.0], dtype=np.float64))
        acc = np.clip(acc, -acc_max, acc_max)

        self._qvel_filt = self._qvel_filt + acc * dt

        # velocity limits (rad/s)
        vel_max = np.deg2rad(np.array([25.0, 25.0, 35.0, 45.0], dtype=np.float64))
        self._qvel_filt = np.clip(self._qvel_filt, -vel_max, vel_max)
        self._qpos_filt = self._clip_joint_vec(self._qpos_filt + self._qvel_filt * dt)
        
        # Carrier motion proposal
        prev_xy = phys.data.qpos[self._carrier_qpos_adr].copy()
        dx_nominal = self._advance_carrier_nominal(dt)

        # Solve keepout without rollback
        carrier_xy = self._solve_carrier_xy(prev_xy, dx_nominal, self._qpos_filt, dt)

        # Final state write
        self._walk_xy = carrier_xy.copy()
        self._set_kinematic_state(carrier_xy, self._qpos_filt)