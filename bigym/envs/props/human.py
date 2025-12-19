"""Human props."""
from pathlib import Path


from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import KinematicProp
import numpy as np
from bigym.envs.props.human_utils import *
from dataclasses import dataclass
from typing import Optional
from dm_control import mjcf

@dataclass
class HumanMotion:
    """Container for a pre-exported motion."""
    joints: np.ndarray        # (T, Nj, 3) MuJoCo/world coords
    bone_pairs: np.ndarray    # (Nb, 2) indices into Nj
    fps: float

class Human(KinematicProp):
    """Human model for interaction with props."""
    _HUMAN_JOINT_XML_DIR: str = ASSETS_PATH / "props/kitchen/base_cabinet_600_with_human.xml"
    _MOTION_JOINTS_DIR: str = ASSETS_PATH / "props/kitchen/banana_eat_1_mujoco_joints.npz"
    _MOTION_JOINTS: np.ndarray = np.load(_MOTION_JOINTS_DIR)  # (T, Nj, 3)
    _MOTION_FPS: float = 30.0 #TODO
    _t:float = 0.0
    _frame:int = 0
    _base_pos: np.ndarray = np.array([0.0, 0.0, 0.0])
    _RESET_WITH_SEED: bool = True
    _SEED: Optional[int] = None

    @property
    def _model_path(self) -> Path:
        return self._HUMAN_JOINT_XML_DIR

    @property
    def _motion_path(self) -> np.ndarray:
        return self._MOTION_JOINTS_DIR

    @property
    def fps(self) -> float:
        """Get motion fps."""
        return self._MOTION_FPS
    
    def set_fps(self, fps: float):
        """Set motion fps."""
        self._MOTION_FPS = fps
    
    def set_reset_with_seed(self, enable: bool):
        """Set whether to randomize reset time with seed."""
        self._RESET_WITH_SEED = enable
    
    def set_seed(self, seed: Optional[int]):
        """Set whether to randomize reset time with seed."""
        self._RESET_WITH_SEED = True if seed is not None else False
        self._SEED = seed

    def _time_to_frame(self, time: float) -> int:
        """Convert time to frame index."""
        return int(time * float(self._MOTION_FPS)) % self._MOTION_JOINTS.shape[0]

    def reset(self, time: float = 0.0, seed: Optional[int] = None):
        """Reset time and frame."""
        self.set_seed(seed)
        if seed is not None:
            t = np.random.Generator(np.random.PCG64(seed)).uniform(0, self._MOTION_JOINTS.shape[0] / float(self._MOTION_FPS)) if seed is not None else 0.0
        else:            
            t = np.random.Generator(np.random.PCG64(self._SEED)).uniform(0, self._MOTION_JOINTS.shape[0] / float(self._MOTION_FPS)) if self._RESET_WITH_SEED else time
        self._t = t
        self._frame = self._time_to_frame(t)
        self._apply_frame(self._frame)

    def step(self, dt: float):
        """Advance time and write mocap poses."""
        self._t += dt
        self._frame = self._time_to_frame(self._t)
        self._apply_frame(self._frame)

    def _apply_frame(self, frame: int):
        joints = self._MOTION_JOINTS[frame].astype(np.float64)  # (Nj,3)
        bone_pairs = self._MOTION_JOINTS.bone_pairs

        # Apply global base transform (translate only, optionally rotate if needed later)
        # For now: base_pos offset is the main practical need.
        joints_w = joints + self._base_pos[None, :]

        physics = self._mojo.physics
        model = physics.model
        data = physics.data

        # Resolve mocap ids once (cache on first call)
        if not hasattr(self, "_joint_body_ids"):
            self._joint_body_ids = []
            Nj = joints_w.shape[0]
            for j in range(Nj):
                bname = f"J{j}"
                try:
                    bid = mjcf.get_attachment_frame(self.body.mjcf).root.model.body(bname)  # may not work in all stacks
                    # Fallback: we'll map via mujoco name2id at runtime below
                    self._joint_body_ids.append(bname)
                except Exception:
                    self._joint_body_ids.append(bname)

            self._bone_body_names = [f"B{b}" for b in range(bone_pairs.shape[0])]

        # MuJoCo native mapping (robust): use mujoco.mj_name2id via dm_control
        # dm_control exposes mujoco module inside physics; safest: use model.name2id if available.
        def _name2bodyid(n: str) -> int:
            # dm_control physics has named accessors in many versions, but not all.
            # This is robust across dm_control builds:
            try:
                return model.name2id(n, "body")
            except Exception:
                import mujoco
                return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # 1) Joint markers (optional, if your fragment has J* bodies)
        for j, name in enumerate(self._joint_body_ids):
            bid = _name2bodyid(name)
            mid = int(model.body_mocapid[bid])
            if mid >= 0:
                data.mocap_pos[mid] = joints_w[j]

        # 2) Bones (capsules) bodies B*
        for b, name in enumerate(self._bone_body_names):
            a, c = bone_pairs[b]
            pa = joints_w[a]
            pb = joints_w[c]
            d = pb - pa
            d_unit = normalize(d)
            q = quat_from_two_unit_vectors(z_axis, d_unit)  # (w,x,y,z)
            center = 0.5 * (pa + pb)

            bid = _name2bodyid(name)
            mid = int(model.body_mocapid[bid])
            if mid >= 0:
                data.mocap_pos[mid] = center
                data.mocap_quat[mid] = q
