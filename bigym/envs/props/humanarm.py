from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import mujoco

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import KinematicProp


ControlMode = Literal["scripted", "position", "torque"]


@dataclass
class PDGains:
    kp: np.ndarray
    kd: np.ndarray


class HumanArm(KinematicProp):
    """
    Kinematic cylinder-arm proxy with:
      - scripted joint motion (smooth, band-limited sine noise)
      - 2D OU wandering of the carrier body (arm_tx, arm_ty)
      - keepout correction to prevent deep penetration into robot geometry

    Design intent:
      - The arm should keep moving (human unaware of robot).
      - We avoid explosive contacts by NOT relying on MuJoCo contact resolution.
      - Instead, we enforce a geometric "keepout" in XY on the carrier.
    """

    _ARM_XML: Path = ASSETS_PATH / "props/human_arm/arm_two_joints.xml"
    _ROOT_PREFIX: str = "cylinder_arm"

    # Minimum allowed clearance between arm capsule and robot keepout spheres (meters).
    MIN_CLEAR: float = 0.02  # 2 cm

    # Keepout sphere radius scale factor: 1.0 is conservative (bigger keepout).
    # If you see the arm stopping too often -> reduce to 0.6~0.8.
    KEEPOUT_SCALE: float = 0.75

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

    _GAINS: PDGains = PDGains(
        kp=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        kd=np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float64),
    )
    _CTRL_CLIP: float = 1.0

    _CURRENT_TIME: float = 0.0
    _MOTION_FPS: float = 30.0

    _NOISE_HARMONICS: int = 4
    _NOISE_FREQ_RANGE: Tuple[float, float] = (0.05, 0.6)  # Hz
    _noise_resample_period: float = 6.0
    _next_resample_t: float = 0.0

    # Planar carrier joints
    _CARRIER_JOINTS: Tuple[str, str] = ("arm_tx", "arm_ty")

    # OU wandering parameters
    _walk_enable: bool = True
    _walk_radius: float = 0.25
    _walk_tau: float = 1.5
    _walk_sigma: float = 0.25
    _walk_speed: float = 0.2  # interpreted as max speed

    def __init__(self, mojo, kinematic=None, cache_colliders=None, cache_sites=None, parent=None, **kwargs):
        super().__init__(mojo, kinematic, cache_colliders, cache_sites, parent, **kwargs)

        self._physics = self._mojo.physics
        m = self._physics.model.ptr
        d = self._physics.data

        # Resolve ids/addresses for arm joints & actuators
        self._act_id = np.array([self._get_actuator_id(n) for n in self._ACT_NAMES], dtype=np.int32)
        self._qpos_adr = np.array([self._get_joint_qpos_adr(n) for n in self._JOINT_NAMES], dtype=np.int32)
        self._qvel_adr = np.array([self._get_joint_dof_adr(n) for n in self._JOINT_NAMES], dtype=np.int32)

        # Resolve carrier planar joints
        self._carrier_qpos_adr = np.array([self._get_joint_qpos_adr(n) for n in self._CARRIER_JOINTS], dtype=np.int32)
        self._carrier_qvel_adr = np.array([self._get_joint_dof_adr(n) for n in self._CARRIER_JOINTS], dtype=np.int32)

        # Infer ctrl clip from actuator ctrlrange
        try:
            cr = self._physics.model.actuator_ctrlrange[self._act_id, :]
            self._CTRL_CLIP = float(np.max(np.abs(cr)))
        except Exception:
            pass

        # RNG + scripted noise params
        self._rng = np.random.default_rng()
        self._noise_freqs = None
        self._noise_phases = None
        self._noise_amps = None

        # External control state
        self._mode: ControlMode = "scripted"
        self._qpos_target = np.zeros(len(self._JOINT_NAMES), dtype=np.float64)
        self._ctrl_target = np.zeros(len(self._ACT_NAMES), dtype=np.float64)

        # Low-pass filter state for kinematic qpos
        self._qpos_filt = d.qpos[self._qpos_adr].copy()

        # OU wander state (in local XY)
        self._walk_xy = np.zeros(2, dtype=np.float64)
        self._walk_v = np.zeros(2, dtype=np.float64)

        # Carrier center (world XY baseline)
        self._walk_center_xy = d.qpos[self._carrier_qpos_adr].copy()

        # Arm geoms (capsule proxy built from geom_xpos/xmat/size in _on_step)
        self._arm_geom_ids = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, self._pref("upperarm_geom")),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, self._pref("forearm_geom")),
        ], dtype=np.int32)
        if np.any(self._arm_geom_ids < 0):
            raise RuntimeError(f"Arm geom ids not found: {self._arm_geom_ids} (check name prefixing)")

        # Robot keepout geoms: collidable + not tiny
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

        # Cache keepout radii (scaled bounding spheres)
        self._robot_keepout_r = self.KEEPOUT_SCALE * np.asarray(m.geom_rbound[self._robot_geom_ids], dtype=np.float64)

        # For video/frame timing (kept from your earlier code)
        self.dt_frame = 1.0 / max(float(self._MOTION_FPS), 1e-9)

    # --------------------
    # Required by Prop API
    # --------------------
    @property
    def _model_path(self) -> Path:
        return self._ARM_XML

    # --------------------
    # Name helpers
    # --------------------
    def _pref(self, name: str) -> str:
        return f"{self._ROOT_PREFIX}/{name}"

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
    # Scripted trajectory
    # --------------------
    def _init_noise(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        num_joints = len(self._JOINT_NAMES)
        harmonics = int(self._NOISE_HARMONICS)

        freqs = self._rng.uniform(self._NOISE_FREQ_RANGE[0], self._NOISE_FREQ_RANGE[1], size=(num_joints, harmonics))
        phases = self._rng.uniform(0.0, 2.0 * np.pi, size=(num_joints, harmonics))
        amps = self._rng.uniform(0.25, 1.0, size=(num_joints, harmonics))
        amps = amps / np.sum(amps, axis=1, keepdims=True)

        self._noise_freqs = freqs
        self._noise_phases = phases
        self._noise_amps = amps

    def _traj(self, t: float) -> np.ndarray:
        if self._noise_freqs is None:
            self._init_noise()

        sinusoids = np.sin(2.0 * np.pi * self._noise_freqs * t + self._noise_phases)
        noise = np.sum(self._noise_amps * sinusoids, axis=1)

        base  = np.deg2rad(30.0) * noise[0]
        yaw   = np.deg2rad(30.0) * noise[1]
        pitch = np.deg2rad(-20.0) + np.deg2rad(20.0) * noise[2]
        elbow = np.deg2rad(60.0)  + np.deg2rad(40.0) * noise[3]
        return np.array([base, yaw, pitch, elbow], dtype=np.float64)

    # --------------------
    # Episode lifecycle
    # --------------------
    def reset(self, seed=None, time: float = 0.0, qpos0: Optional[np.ndarray] = None, mode: Optional[ControlMode] = None):
        self._CURRENT_TIME = float(time)
        self._init_noise(seed)
        if mode is not None:
            self._mode = mode

        if qpos0 is None:
            qpos0 = self._traj(self._CURRENT_TIME)
        else:
            qpos0 = np.asarray(qpos0, dtype=np.float64).reshape(4,)

        d = self._physics.data
        d.qpos[self._qpos_adr] = qpos0
        d.qvel[self._qvel_adr] = 0.0
        d.ctrl[self._act_id] = 0.0

        self._qpos_target[:] = d.qpos[self._qpos_adr]
        self._ctrl_target[:] = 0.0
        self._qpos_filt = d.qpos[self._qpos_adr].copy()

        d.qpos[self._carrier_qpos_adr] = 0.0
        d.qvel[self._carrier_qvel_adr] = 0.0
        self._walk_center_xy = np.zeros(2, dtype=np.float64)
        self._walk_xy = np.zeros(2, dtype=np.float64)
        self._walk_v = np.zeros(2, dtype=np.float64)

        self._next_resample_t = self._CURRENT_TIME + self._noise_resample_period
        self._physics.forward()

    # --------------------
    # Core step
    # --------------------
    def _on_step(self, dt: float):
        """
        Fast keepout step:

        1) Update carrier OU wander (XY).
        2) Update arm joint target (scripted/position), apply kinematic override.
        3) Forward once.
        4) If arm too close to robot keepout spheres:
            - remove inward component of carrier velocity
            - add a small push-out along normal (<= 2cm per step)
            - forward once more
            - if still violating, revert carrier (not joints) and damp velocity

        This prevents deep penetration without freezing joints (reduces deadlocks).
        """

        self._CURRENT_TIME += float(dt)
        dt = float(dt)

        phys = self._physics
        m = phys.model.ptr
        d = phys.data.ptr

        # Save previous carrier state for rejection
        carrier_prev = phys.data.qpos[self._carrier_qpos_adr].copy()
        walk_xy_prev = self._walk_xy.copy()
        walk_v_prev = self._walk_v.copy()

        # -------------------------
        # 1) OU carrier update
        # -------------------------
        if self._walk_enable:
            tau = max(float(self._walk_tau), 1e-6)
            vmax = float(self._walk_speed)
            sigma = float(self._walk_sigma)

            # OU on velocity
            self._walk_v += (-self._walk_v / tau) * dt + sigma * np.sqrt(dt) * self._rng.standard_normal(2)

            vnorm = float(np.linalg.norm(self._walk_v))
            if vnorm > vmax:
                self._walk_v = self._walk_v / (vnorm + 1e-12) * vmax

            # integrate
            self._walk_xy = self._walk_xy + self._walk_v * dt

            # stay within disk
            r = float(self._walk_radius)
            rad = float(np.linalg.norm(self._walk_xy))
            if rad > r:
                self._walk_xy = self._walk_xy / (rad + 1e-12) * r
                outward = self._walk_xy / (float(np.linalg.norm(self._walk_xy)) + 1e-12)
                self._walk_v -= outward * max(0.0, float(np.dot(self._walk_v, outward)))

        carrier_prop = self._walk_center_xy + self._walk_xy

        # -------------------------
        # 2) Joints update
        # -------------------------
        if self._mode == "torque":
            u = self._ctrl_target
            clip = float(self._CTRL_CLIP)
            if clip > 0:
                u = np.clip(u, -clip, clip)
            phys.data.ctrl[self._act_id] = u
            phys.forward()
            return

        if self._CURRENT_TIME >= self._next_resample_t:
            self._init_noise()
            self._next_resample_t += self._noise_resample_period

        if self._mode == "scripted":
            q_des = self._traj(self._CURRENT_TIME)
            self._qpos_target[:] = q_des
        elif self._mode == "position":
            q_des = self._qpos_target
        else:
            raise RuntimeError(f"Unknown control mode: {self._mode}")

        # low-pass for continuity (time constant ~0.25s)
        alpha = 1.0 - np.exp(-dt / 0.25)
        self._qpos_filt = (1.0 - alpha) * self._qpos_filt + alpha * q_des

        # -------------------------
        # 3) Apply + forward (once)
        # -------------------------
        phys.data.qpos[self._carrier_qpos_adr] = carrier_prop
        phys.data.qvel[self._carrier_qvel_adr] = 0.0

        phys.data.qpos[self._qpos_adr] = self._qpos_filt
        phys.data.qvel[self._qvel_adr] = 0.0
        phys.data.ctrl[self._act_id] = 0.0

        phys.forward()

        # -------------------------
        # 4) Keepout check (arm capsules vs robot keepout spheres)
        # -------------------------
        keep_ids = self._robot_geom_ids
        if keep_ids is None or len(keep_ids) == 0 or (not self._walk_enable):
            return

        # Robot centers + cached radii
        X = np.asarray(d.geom_xpos[keep_ids], dtype=np.float64)      # (N,3)
        Rb = self._robot_keepout_r                                   # (N,)

        best_clear = float("inf")
        best_nxy = None

        # For each arm segment, compute min clearance to all robot spheres (vectorized over robot geoms)
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

            # Closest points on segment to each robot sphere center
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
                n = -V[j] / (float(dist_center[j]) + 1e-12)  # push direction (world)
                best_nxy = n[:2].copy()                       # only carrier XY

        # If safe, we’re done
        if best_clear >= self.MIN_CLEAR or best_nxy is None:
            return

        # -------------------------
        # 5) Tangential slide + small push-out (one extra forward max)
        # -------------------------
        nxy = best_nxy
        nrm = float(np.linalg.norm(nxy))
        if nrm < 1e-9:
            # degenerate normal, reject carrier
            phys.data.qpos[self._carrier_qpos_adr] = carrier_prev
            phys.data.qvel[self._carrier_qvel_adr] = 0.0
            self._walk_xy = walk_xy_prev
            self._walk_v = np.zeros_like(self._walk_v)
            phys.forward()
            return

        nxy /= nrm

        # Remove inward velocity component (prevents pushing further into robot)
        vn = float(np.dot(self._walk_v, nxy))
        if vn < 0.0:
            self._walk_v = self._walk_v - vn * nxy

        # Add a limited push-out so we don’t "stop" or jitter at boundary
        push = float(np.clip(self.MIN_CLEAR - best_clear, 0.0, 0.02))  # <= 2cm/step
        self._walk_xy = walk_xy_prev + self._walk_v * dt + push * nxy

        # Keep within wander radius disk
        r = float(self._walk_radius)
        rad = float(np.linalg.norm(self._walk_xy))
        if rad > r:
            self._walk_xy = self._walk_xy / (rad + 1e-12) * r

        carrier_fix = self._walk_center_xy + self._walk_xy
        phys.data.qpos[self._carrier_qpos_adr] = carrier_fix
        phys.data.qvel[self._carrier_qvel_adr] = 0.0
        phys.forward()

        # -------------------------
        # 6) Cheap recheck once; if still violating, reject carrier
        # -------------------------
        X = np.asarray(d.geom_xpos[keep_ids], dtype=np.float64)
        best2 = float("inf")

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
            best2 = min(best2, float(np.min(clear)))

        if best2 < self.MIN_CLEAR:
            # Reject carrier only; keep joints moving
            phys.data.qpos[self._carrier_qpos_adr] = carrier_prev
            phys.data.qvel[self._carrier_qvel_adr] = 0.0
            self._walk_xy = walk_xy_prev
            self._walk_v = 0.2 * walk_v_prev  # damp, but not zero
            phys.forward()