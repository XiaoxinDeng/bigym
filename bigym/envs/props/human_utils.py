import numpy as np
import mujoco

def quat_from_z_to_vec(v):
    # returns quaternion rotating +Z to v (v need not be normalized)
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v = v / n
    z = np.array([0.0, 0.0, 1.0])
    c = np.dot(z, v)

    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        # 180 deg rotation about X axis (any axis orthogonal to z works)
        return np.array([0.0, 1.0, 0.0, 0.0])

    axis = np.cross(z, v)
    s = np.sqrt((1.0 + c) * 2.0)
    q = np.array([s * 0.5, axis[0] / s, axis[1] / s, axis[2] / s])
    return q

def set_mocap_body(model, data, body_name, pos, quat=(1,0,0,0)):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mid = model.body_mocapid[bid]
    if mid < 0:
        raise RuntimeError(f"Body {body_name} is not mocap.")
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = quat

def drive_frame(model, data, joints, bone_pairs):
    # joints: (J,3) for this frame
    # 1) joints as spheres
    for j in range(joints.shape[0]):
        set_mocap_body(model, data, f"J{j}", joints[j], (1,0,0,0))

    # 2) bones as capsules
    for b, (p, c) in enumerate(bone_pairs):
        a = joints[p]
        d = joints[c]
        mid = 0.5 * (a + d)
        v = d - a
        q = quat_from_z_to_vec(v)
        set_mocap_body(model, data, f"B{b}", mid, q)

    mujoco.mj_forward(model, data)

# ----------------------------
# Quaternion utilities
# ----------------------------
def quat_from_two_unit_vectors(u, v, eps=1e-8):
    """
    Returns quaternion q (w, x, y, z) rotating unit vector u -> unit vector v.
    Robust for near-opposite vectors.
    """
    # dot = cos(theta)
    dot = float(np.dot(u, v))
    if dot > 1.0:
        dot = 1.0
    if dot < -1.0:
        dot = -1.0

    if dot > 1.0 - eps:
        # nearly identical
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if dot < -1.0 + eps:
        # nearly opposite: choose an arbitrary orthogonal axis
        # find axis orthogonal to u
        axis = np.cross(u, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < eps:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = axis / (np.linalg.norm(axis) + eps)
        # 180 deg rotation => w=0, xyz=axis
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)

    axis = np.cross(u, v)
    w = 1.0 + dot
    q = np.array([w, axis[0], axis[1], axis[2]], dtype=np.float64)
    q = q / (np.linalg.norm(q) + eps)
    return q


def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def render_video_offscreen(xml_path, npz_path, out_mp4,
                           width=640, height=480,
                           every=1, speed=1.0,
                           distance_mult=1.8,
                           center_on="pelvis", pelvis_index=0):
    """
    Scripted camera without XML camera and without MjrContext.

    Key idea:
      - Center the motion at origin by translating mocap bodies each frame.
      - Set model.stat.center/extents so free camera frames the scene.
      - Use Renderer.update_scene(camera=-1) (free camera).
    """
    import imageio
    import mujoco
    import numpy as np

    writer = None
    renderer = None

    data_npz = np.load(npz_path, allow_pickle=True)
    joints = data_npz["joints"].astype(np.float32)      # (T, Nj, 3) MuJoCo coords
    bone_pairs = data_npz["bone_pairs"].astype(np.int32)
    fps = float(data_npz["fps"])
    out_fps = fps * float(speed) / max(int(every), 1)

    # --- Compute global bbox for stable framing ---
    sample = joints[::max(1, joints.shape[0] // 200)]
    mn = sample.min(axis=(0, 1))
    mx = sample.max(axis=(0, 1))
    center_global = 0.5 * (mn + mx)
    extent = float(np.linalg.norm(mx - mn))  # diagonal length

    # --- Load model/data ---
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Hint MuJoCo about scene scale (affects default free camera framing)
    model.stat.center[:] = center_global
    model.stat.extent = max(0.5, extent)

    # Resolve mocap bodies
    Nj = joints.shape[1]
    Nb = bone_pairs.shape[0]

    joint_body_ids = []
    for j in range(Nj):
        name = f"J{j}"
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            bid = -1
        joint_body_ids.append(bid)

    bone_body_ids = []
    for b in range(Nb):
        name = f"B{b}"
        bone_body_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # --- Initialize renderer & writer ---
    renderer = mujoco.Renderer(model, width=width, height=height)
    writer = imageio.get_writer(out_mp4, fps=out_fps)

    # --- Free camera id is -1 ---
    cam_id = -1

    T = joints.shape[0]
    for frame in range(0, T, max(int(every), 1)):
        Jf = joints[frame]  # (Nj, 3)

        # Choose per-frame center (tracking camera) or fixed center (static)
        if center_on == "pelvis":
            center = Jf[pelvis_index]
        else:
            center = center_global

        # Translate so subject stays near origin
        Jc = Jf - center[None, :]

        # Update joint markers
        for j in range(Nj):
            bid = joint_body_ids[j]
            if bid >= 0:
                mid = model.body_mocapid[bid]
                if mid >= 0:
                    data.mocap_pos[mid] = Jc[j]

        # Update bones
        for b in range(Nb):
            a, c = bone_pairs[b]
            pa = Jc[a].astype(np.float64)
            pb = Jc[c].astype(np.float64)
            d = pb - pa
            n = np.linalg.norm(d)
            if n < 1e-12:
                q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                q = quat_from_two_unit_vectors(z_axis, d / n)

            center_b = 0.5 * (pa + pb)

            bid = bone_body_ids[b]
            mid = model.body_mocapid[bid]
            data.mocap_pos[mid] = center_b
            data.mocap_quat[mid] = q

        mujoco.mj_forward(model, data)

        # Script camera by setting the renderer's internal free camera parameters.
        # In mujoco.Renderer, free camera uses model.stat.*; distance is derived from extent.
        # We can influence distance by temporarily scaling stat.extent.
        model.stat.center[:] = 0.0, 0.0, 0.0
        model.stat.extent = max(0.5, float(extent) * float(distance_mult))

        renderer.update_scene(data, camera=cam_id)
        img = renderer.render()
        writer.append_data(img)

    writer.close()
    renderer.close()
    
