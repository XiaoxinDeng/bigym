"""Human arm prop driven by dm_control actuators (BiGym-compatible), externally controllable."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import KinematicProp


ControlMode = Literal["scripted", "position", "torque"]


@dataclass
class PDGains:
    kp: np.ndarray  # (num_acts,)
    kd: np.ndarray  # (num_acts,)


class HumanArm(KinematicProp):
    """
    Articulated cylinder-arm proxy controllable from outside:

    - mode="scripted": internal trajectory provides desired qpos each step
    - mode="position": external qpos target via set_qpos_target(); internal PD -> ctrl
    - mode="torque":   external ctrl via set_ctrl(); directly writes to data.ctrl
    """

    # --- asset path ---
    _ARM_XML: Path = ASSETS_PATH / "props/human_arm/arm_two_joints.xml"

    # --- name prefixing ---
    _ROOT_PREFIX: str = "cylinder_arm"  # e.g. "base_cabinet_600_with_human/" if names are prefixed

    # --- names (without prefix) ---
    _JOINT_NAMES: Tuple[str, str, str] = ("arm_shoulder_base", "arm_shoulder_yaw", "arm_shoulder_pitch", "arm_elbow")
    _ACT_NAMES:   Tuple[str, str, str] = ("act_arm_shoulder_base", "act_arm_shoulder_yaw", "act_arm_shoulder_pitch", "act_arm_elbow")

    # --- control ---
    _GAINS: PDGains = PDGains(
        kp=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        kd=np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float64),
    )
    _CTRL_CLIP: float = 1.0  # will be overwritten from actuator_ctrlrange if available

    # --- motion time ---
    _CURRENT_TIME: float = 0.0
    _MOTION_FPS: float = 30.0

    # Cached handles
    _physics = None
    _act_id: np.ndarray = None     # (3,) indices into data.ctrl
    _qpos_adr: np.ndarray = None   # (3,) indices into data.qpos
    _qvel_adr: np.ndarray = None   # (3,) indices into data.qvel

    # External control state
    _mode: ControlMode = "scripted"
    _qpos_target: np.ndarray = None   # (3,) radians
    _ctrl_target: np.ndarray = None   # (3,) torque/effort
    _hold_last_target: bool = True    # if True, keeps last target when not updated

    def __init__(self, mojo, kinematic=None, cache_colliders=None, cache_sites=None, parent=None, **kwargs):
        super().__init__(mojo, kinematic, cache_colliders, cache_sites, parent, **kwargs)

        self._physics = self._mojo.physics

        # Resolve indices
        self._act_id = np.array([self._get_actuator_id(n) for n in self._ACT_NAMES], dtype=np.int32)
        self._qpos_adr = np.array([self._get_joint_qpos_adr(n) for n in self._JOINT_NAMES], dtype=np.int32)
        self._qvel_adr = np.array([self._get_joint_dof_adr(n) for n in self._JOINT_NAMES], dtype=np.int32)

        # Infer ctrl clip from model
        try:
            cr = self._physics.model.actuator_ctrlrange[self._act_id, :]  # (3,2)
            self._CTRL_CLIP = float(np.max(np.abs(cr)))
        except Exception:
            pass

        # Default targets
        self._qpos_target = np.zeros(len(self._JOINT_NAMES), dtype=np.float64)
        self._ctrl_target = np.zeros(len(self._ACT_NAMES), dtype=np.float64)

        self.dt_frame = 1.0 / max(float(self._MOTION_FPS), 1e-9)


    # --------------------
    # Required by Prop API
    # --------------------
    @property
    def _model_path(self) -> Path:
        return self._ARM_XML

    # --------------------
    # Name lookup helpers
    # --------------------
    def _pref(self, name: str) -> str:
        return f"{self._ROOT_PREFIX}/{name}"

    def _get_actuator_id(self, act_name: str) -> int:
        m = self._physics.model
        full = self._pref(act_name)
        try:
            return m.actuator(full).id
        except Exception as e:
            raise RuntimeError(f"Cannot find actuator '{full}' in physics.model") from e

    def _get_joint_qpos_adr(self, joint_name: str) -> int:
        m = self._physics.model
        full = self._pref(joint_name)
        try:
            jid = m.joint(full).id
            return int(m.jnt_qposadr[jid])
        except Exception as e:
            raise RuntimeError(f"Cannot find joint '{full}' (qpos adr) in physics.model") from e

    def _get_joint_dof_adr(self, joint_name: str) -> int:
        m = self._physics.model
        full = self._pref(joint_name)
        try:
            jid = m.joint(full).id
            return int(m.jnt_dofadr[jid])
        except Exception as e:
            raise RuntimeError(f"Cannot find joint '{full}' (dof adr) in physics.model") from e

    # --------------------
    # External control API
    # --------------------
    def set_mode(self, mode: ControlMode):
        """
        Switch control mode.
        - scripted: uses internal _traj(t)
        - position: uses externally set qpos target (PD)
        - torque:   uses externally set ctrl target (direct)
        """
        if mode not in ("scripted", "position", "torque"):
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode

    def set_pd_gains(self, kp: np.ndarray, kd: np.ndarray):
        kp = np.asarray(kp, dtype=np.float64).reshape(3,)
        kd = np.asarray(kd, dtype=np.float64).reshape(3,)
        self._GAINS = PDGains(kp=kp, kd=kd)

    def set_qpos_target(self, q_des: np.ndarray, *, radians: bool = True):
        """
        External position target for mode='position'.
        q_des: (3,) in radians by default; set radians=False for degrees.
        """
        q = np.asarray(q_des, dtype=np.float64).reshape(len(self._ACT_NAMES),)
        if not radians:
            q = np.deg2rad(q)
        self._qpos_target[:] = q
        if self._mode != "position":
            self._mode = "position"

    def set_ctrl(self, u: np.ndarray):
        """
        External torque/effort command for mode='torque'.
        u: (3,) will be clipped to ctrlrange.
        """
        u = np.asarray(u, dtype=np.float64).reshape(len(self._ACT_NAMES),)
        self._ctrl_target[:] = u
        if self._mode != "torque":
            self._mode = "torque"

    def get_state(self):
        """Convenience accessor for debugging/obs."""
        d = self._physics.data
        q = d.qpos[self._qpos_adr].copy()
        qd = d.qvel[self._qvel_adr].copy()
        return {"qpos": q, "qvel": qd, "mode": self._mode}

    # --------------------
    # Scripted trajectory (optional)
    # --------------------
    def _traj(self, t: float) -> np.ndarray:
        # radians
        base   = np.deg2rad(30.0) * np.sin(2.0 * np.pi * 0.2 * t)
        yaw   = np.deg2rad(30.0) * np.sin(2.0 * np.pi * 0.2 * t)
        pitch = np.deg2rad(-40.0) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.2 * t + 0.7))
        elbow = np.deg2rad(20.0 + 80.0 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.2 * t + 1.4)))
        return np.array([base, yaw, pitch, elbow], dtype=np.float64)

    # --------------------
    # Episode lifecycle
    # --------------------
    def reset(self, seed=None, time: float = 0.0, qpos0: Optional[np.ndarray] = None, mode: Optional[ControlMode] = None):
        """
        Reset time and optionally pose/mode.
        - If qpos0 provided: sets joint pose.
        - Else: uses scripted trajectory pose at `time`.
        """
        self._CURRENT_TIME = float(time)

        if mode is not None:
            self.set_mode(mode)

        if qpos0 is None:
            qpos0 = self._traj(self._CURRENT_TIME)
        else:
            qpos0 = np.asarray(qpos0, dtype=np.float64).reshape(3,)

        d = self._physics.data
        d.qpos[self._qpos_adr] = qpos0
        d.qvel[self._qvel_adr] = 0.0
        d.ctrl[self._act_id] = 0.0

        # Initialize targets to current pose (prevents a jump on first step in position mode)
        self._qpos_target[:] = d.qpos[self._qpos_adr]
        self._ctrl_target[:] = 0.0

        self._physics.forward()

    def _on_step(self, dt: float):
        """
        Called by BiGym each step. Writes controls according to selected mode.
        """
        self._CURRENT_TIME += float(dt)
        d = self._physics.data

        if self._mode == "torque":
            u = self._ctrl_target
            clip = float(self._CTRL_CLIP)
            if clip > 0:
                u = np.clip(u, -clip, clip)
            d.ctrl[self._act_id] = u
            return

        # Determine desired qpos
        if self._mode == "scripted":
            q_des = self._traj(self._CURRENT_TIME)
            # Optionally expose the current scripted target as "last target"
            self._qpos_target[:] = q_des
        elif self._mode == "position":
            q_des = self._qpos_target
        else:
            raise RuntimeError(f"Unknown control mode: {self._mode}")

        # PD -> torque
        q  = d.qpos[self._qpos_adr]
        qd = d.qvel[self._qvel_adr]
        kp = self._GAINS.kp
        kd = self._GAINS.kd

        u = kp * (q_des - q) - kd * qd

        clip = float(self._CTRL_CLIP)
        if clip > 0:
            u = np.clip(u, -clip, clip)

        d.ctrl[self._act_id] = u
