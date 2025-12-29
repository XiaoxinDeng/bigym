"""Human props."""
from pathlib import Path


from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import KinematicProp
import numpy as np
from bigym.envs.props.human_utils import *
from dataclasses import dataclass
from typing import Optional
from dm_control import mjcf
from bigym.utils.quaternion import *

class Human(KinematicProp):
    """Human model for interaction with props."""
    _HUMAN_JOINT_XML_DIR: str = ASSETS_PATH / "props/kitchen/base_cabinet_600_with_human_apple_eat.xml"
    _MOTION_JOINTS_DIR: str = ASSETS_PATH / "props/kitchen/apple_eat_1_mujoco_joints.npz"
    _MOTION_FILE: np.ndarray = np.load(_MOTION_JOINTS_DIR)  # (T, Nj, 3)
    _MOTION_FPS: float = 30.0 
    _CURRENT_TIME:float = 0.0
    _CURRENT_FRAME:int = 0
    _RESET_WITH_SEED: bool = True
    _SEED: Optional[int] = None
    _PARENTS: Optional[np.ndarray] = None
    _NUM_FRAMES: int = 0
    _NUM_JOINTS:int = 0
    
    _ROOT_PREFIX:str = "base_cabinet_600_with_human/"
    _BALL_JOINT_PREFIX: str = _ROOT_PREFIX+"joint_"
    # qpos attributes
    _qpos_root_t: np.ndarray = None
    _qpos_root_r: np.ndarray = None
    _qpos_root_tx: np.ndarray = None
    _qpos_root_ty: np.ndarray = None
    _qpos_root_tz: np.ndarray = None
    _qpos_root_rx: np.ndarray = None
    _qpos_root_ry: np.ndarray = None
    _qpos_root_rz: np.ndarray = None
    _qpos_ball: np.ndarray = None

    def __init__(self, mojo, kinematic = None, cache_colliders = None, cache_sites = None, parent = None, **kwargs):
        super().__init__(mojo, kinematic, cache_colliders, cache_sites, parent, **kwargs)
        
        if "joints" not in self._MOTION_FILE or "fps" not in self._MOTION_FILE:
            raise RuntimeError("self._MOTION_FILE must contain 'joints' and 'fps'.")
        if self._MOTION_JOINTS.ndim != 3 or self._MOTION_JOINTS.shape[2] != 3:
            raise RuntimeError(f"self._MOTION_FILE 'joints' must have shape (T, Nj, 3), got {self._MOTION_JOINTS.shape}")
        
        self._MOTION_JOINTS = self._MOTION_FILE['joints'].astype(np.float64)
        self._MOTION_FPS = float(self._MOTION_FILE['fps'])
        self.dt_frame = 1.0 / max(self._MOTION_FPS, 1e-9)
        self._NUM_FRAMES, self._NUM_JOINTS, _ = self._MOTION_JOINTS.shape
        
        # parents (preferred)
        if "parents" in self._MOTION_FILE:
            self._PARENTS = self._MOTION_FILE["parents"].astype(np.int32)
            if self._PARENTS.shape != (self._NUM_JOINTS,):
                raise RuntimeError(f"self._MOTION_FILE 'parents' must have shape ({self._NUM_JOINTS},), got {self._PARENTS.shape}")
        elif "bone_pairs" in self._MOTION_FILE:
            self._PARENTS = np.full((self._NUM_JOINTS,), -1, dtype=np.int32)
            bp = self._MOTION_FILE["bone_pairs"].astype(np.int32)
            for p, c in bp:
                if 0 <= c < self._NUM_JOINTS:
                    self._PARENTS[c] = int(p)
            if self._PARENTS[0] != -1:
                self._PARENTS[0] = -1
        else:
            raise RuntimeError("self._MOTION_FILE must contain 'parents' or 'bone_pairs' to define the kinematic tree.")
        
        physics = self._mojo.physics
        self._qpos_root_t = np.zeros(3, dtype=np.float64)
        self._qpos_root_r = np.zeros(3, dtype=np.float64)
        self._qpos_root_tx = self._get_qpos_addr(physics, self._ROOT_PREFIX+"root_tx")  # scalar view
        self._qpos_root_ty = self._get_qpos_addr(physics, self._ROOT_PREFIX+"root_ty")
        self._qpos_root_tz = self._get_qpos_addr(physics, self._ROOT_PREFIX+"root_tz")
        self._qpos_root_rx = self._get_qpos_addr(physics, self._ROOT_PREFIX+"root_rx")
        self._qpos_root_ry = self._get_qpos_addr(physics, self._ROOT_PREFIX+"root_ry")
        self._qpos_root_rz = self._get_qpos_addr(physics, self._ROOT_PREFIX+"root_rz")
        
        self._qpos_ball = [None] * self._NUM_JOINTS
        for i in range(1, self._NUM_JOINTS):
            self._qpos_ball[i] = self._get_qpos_addr(physics, f"{self._BALL_JOINT_PREFIX}{i}")
    
    def _get_qpos_addr(self, physics, name:str):
        """
        Safely obtain qpos address from model
        """
        try:
            return physics.named.data.qpos[name]
        except KeyError as e:
            raise RuntimeError(f"Cannot find joint '{name}' in physics.named.data.qpos") from e

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
        self._CURRENT_TIME = t
        self._CURRENT_FRAME = self._time_to_frame(t)
        self._apply_frame(self._CURRENT_FRAME)

    def step(self, dt: float):
        """Advance time and write mocap poses."""
        self._CURRENT_TIME += dt
        self._CURRENT_FRAME = self._time_to_frame(self._CURRENT_TIME)
        self._apply_frame(self._CURRENT_FRAME)

    def _apply_frame(self, frame: int):
        physics = self._mojo.physics
        pelvis_index: int = 0
        Jw:np.ndarray = self._MOTION_JOINTS[frame].astype(np.float64)  # (Nj,3) world
        Nj:int = self._NUM_JOINTS
        parents:np.ndarray = self._PARENTS
        bone_rest_axis:tuple=(0.0, 0.0, 1.0)
        center_world:np.ndarray = Jw[int(pelvis_index)]
        
        # set root position
        self._qpos_root_tx[...] = center_world[0]
        self._qpos_root_ty[...] = center_world[1]
        self._qpos_root_tz[...] = center_world[2]
        # set root orientation
        self._qpos_root_rx[...] = 0.0
        self._qpos_root_ry[...] = 0.0
        self._qpos_root_rz[...] = 0.0
        # ----------------------------
        # Choose world-space center (root translation)
        # ----------------------------
        
        # For rotation computation, centering is optional; it does not change bone directions.
        Jc = Jw - center_world[None, :]

        rest_axis = normalize(np.array(bone_rest_axis, dtype=np.float64))

        # ----------------------------
        # Compute local joint quats
        # ----------------------------
        q_world = {0: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)}  # assume root identity
        q_local = [None] * Nj
        q_local[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for i in range(1, Nj):
            p = int(parents[i])
            if p < 0:
                p = 0

            d_world = Jc[i] - Jc[p]
            if np.linalg.norm(d_world) < 1e-10:
                qloc = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                d_world = normalize(d_world)
                q_p_world = q_world.get(p, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
                d_parent = rotate_vec_by_quat(quat_inv(q_p_world), d_world)
                qloc = quat_from_two_unit_vectors(rest_axis, normalize(d_parent))

            q_local[i] = qloc
            q_world[i] = quat_mul(q_world.get(p, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)), qloc)

        # ----------------------------
        # Write qpos via cached named views
        # ----------------------------
        
        # ball joints qpos: [qw,qx,qy,qz]
        for i in range(1, Nj):
            self._qpos_ball[i][:] = q_local[i]

        physics.forward()
