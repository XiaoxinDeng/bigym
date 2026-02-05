"""Human props with mocap skeleton."""
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Geom

from bigym.const import ASSETS_PATH
from bigym.envs.props.human_utils import normalize, quat_from_two_unit_vectors


@dataclass
class HumanMotion:
    """Container for a pre-exported motion."""
    joints: np.ndarray        # (T, Nj, 3) MuJoCo/world coords
    bone_pairs: np.ndarray    # (Nb, 2) indices into Nj
    fps: float


class Human:
    """Human skeleton with mocap bodies for motion playback.

    This class directly adds mocap bodies to worldbody (required by MuJoCo)
    rather than using the standard Prop loading mechanism.
    """

    _instance_counter: int = 0
    _MOTION_NPZ_PATH: Path = ASSETS_PATH / "props/kitchen/banana_eat_1_mujoco_joints.npz"
    _DEFAULT_FPS: float = 30.0

    def __init__(
        self,
        mojo: Mojo,
        motion_path: Optional[Path] = None,
        fps: Optional[float] = None,
    ):
        """Initialize human skeleton."""
        self._mojo = mojo
        self._instance_id = Human._instance_counter
        Human._instance_counter += 1

        # Load motion data
        motion_path = motion_path or self._MOTION_NPZ_PATH
        motion_data = np.load(str(motion_path), allow_pickle=True)
        self._motion_joints = motion_data['joints']  # (T, Nj, 3)
        self._bone_pairs = motion_data['bone_pairs']  # (Nb, 2)
        self._fps = fps or self._DEFAULT_FPS

        self._num_joints = self._motion_joints.shape[1]
        self._num_bones = self._bone_pairs.shape[0]
        self._num_frames = self._motion_joints.shape[0]

        # Time tracking
        self._t: float = 0.0
        self._frame: int = 0
        self._base_pos: np.ndarray = np.array([0.0, 0.0, 0.0])

        # Reset behavior
        self._reset_with_seed: bool = True
        self._seed: Optional[int] = None

        # Add mocap bodies to worldbody
        self._joint_names: List[str] = []
        self._bone_names: List[str] = []
        self._add_mocap_bodies_to_worldbody()

        # Cache for collision checking
        self.colliders: List[Geom] = []
        self._cache_colliders()

    def _get_unique_name(self, base: str) -> str:
        """Generate unique name for this human instance."""
        if self._instance_id == 0:
            return base
        return f"{base}_h{self._instance_id}"

    def _add_mocap_bodies_to_worldbody(self):
        """Add mocap bodies directly to worldbody via MJCF."""
        root_mjcf = self._mojo.root_element.mjcf
        worldbody = root_mjcf.worldbody

        # Add joint markers (spheres)
        for j in range(self._num_joints):
            name = self._get_unique_name(f"J{j}")
            self._joint_names.append(name)
            body = worldbody.add('body', name=name, mocap='true')
            body.add('geom', type='sphere', size=[0.01],
                     rgba=[0.2, 0.6, 1.0, 0.7], contype=0, conaffinity=0)

        # Add bone capsules
        for b in range(self._num_bones):
            name = self._get_unique_name(f"B{b}")
            self._bone_names.append(name)
            body = worldbody.add('body', name=name, mocap='true')
            body.add('geom', name=self._get_unique_name(f"bone{b}"),
                     type='capsule', size=[0.035, 0.05],
                     contype=2, conaffinity=1, rgba=[1.0, 0.6, 0.6, 0.6])

        self._mojo.mark_dirty()

    def _cache_colliders(self):
        """Cache bone colliders for collision checking."""
        physics = self._mojo.physics
        self.colliders = []
        for idx in range(len(self._bone_names)):
            geom_name = self._get_unique_name(f"bone{idx}")
            try:
                geom_mjcf = self._mojo.root_element.mjcf.find('geom', geom_name)
                if geom_mjcf is not None:
                    self.colliders.append(Geom(self._mojo, geom_mjcf))
            except Exception:
                pass

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def num_joints(self) -> int:
        return self._num_joints

    def set_fps(self, fps: float):
        self._fps = fps

    def set_seed(self, seed: Optional[int]):
        self._reset_with_seed = seed is not None
        self._seed = seed

    def _time_to_frame(self, time: float) -> int:
        return int(time * self._fps) % self._num_frames

    def reset(self, time: float = 0.0, seed: Optional[int] = None):
        """Reset time and frame."""
        self.set_seed(seed)
        if seed is not None:
            rng = np.random.Generator(np.random.PCG64(seed))
            t = rng.uniform(0, self._num_frames / self._fps)
        elif self._reset_with_seed and self._seed is not None:
            rng = np.random.Generator(np.random.PCG64(self._seed))
            t = rng.uniform(0, self._num_frames / self._fps)
        else:
            t = time
        self._t = t
        self._frame = self._time_to_frame(t)
        self._apply_frame(self._frame)

    def step(self, dt: float):
        """Advance time and update mocap poses."""
        self._t += dt
        self._frame = self._time_to_frame(self._t)
        self._apply_frame(self._frame)

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions as flattened array.

        Returns:
            np.ndarray: Shape (num_joints * 3,) - flattened (x,y,z) for each joint
        """
        joints = self._motion_joints[self._frame].astype(np.float32)
        return (joints + self._base_pos[None, :]).flatten()

    def _apply_frame(self, frame: int):
        """Apply motion frame to mocap bodies."""
        joints = self._motion_joints[frame].astype(np.float64)
        joints_w = joints + self._base_pos[None, :]

        physics = self._mojo.physics
        model = physics.model
        data = physics.data
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Update joint markers
        for j, name in enumerate(self._joint_names):
            try:
                bid = model.name2id(name, 'body')
                mid = int(model.body_mocapid[bid])
                if mid >= 0:
                    data.mocap_pos[mid] = joints_w[j]
            except Exception:
                pass

        # Update bone capsules
        for b, name in enumerate(self._bone_names):
            try:
                a, c = self._bone_pairs[b]
                pa, pb = joints_w[a], joints_w[c]
                d = pb - pa
                d_unit = normalize(d)
                q = quat_from_two_unit_vectors(z_axis, d_unit)
                bid = model.name2id(name, 'body')
                mid = int(model.body_mocapid[bid])
                if mid >= 0:
                    data.mocap_pos[mid] = 0.5 * (pa + pb)
                    data.mocap_quat[mid] = q
            except Exception:
                pass

    def is_colliding(self, other) -> bool:
        """Check collision with another prop or geom."""
        from bigym.utils.physics_utils import has_collided_collections, get_colliders
        other_colliders = get_colliders(other)
        return has_collided_collections(self._mojo.physics, self.colliders, other_colliders)

    def get_pose(self) -> np.ndarray:
        return self._base_pos.copy()

    def set_pose(self, position: np.ndarray):
        self._base_pos = position.copy()
        self._apply_frame(self._frame)
