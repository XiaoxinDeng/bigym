import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("base_cabinet_600_with_human.xml")
data  = mujoco.MjData(model)

# Make sure kinematics are computed so geom_xpos is valid
mujoco.mj_forward(model, data)

def compute_scene_bounds_world(model, data):
    # data.geom_xpos is (ngeom, 3) in world frame
    xpos = data.geom_xpos.copy()

    # model.geom_rbound is a conservative bounding-sphere radius per geom
    r = model.geom_rbound.reshape(-1, 1)

    mn = (xpos - r).min(axis=0)
    mx = (xpos + r).max(axis=0)

    center = 0.5 * (mn + mx)
    extent = float(np.linalg.norm(mx - mn))
    return center, extent, mn, mx

center, extent, mn, mx = compute_scene_bounds_world(model, data)
print("World bounds min:", mn, "max:", mx)
print("Center:", center, "Extent:", extent)

with mujoco.viewer.launch(model, data) as viewer:
    # Force free camera (important)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Put camera somewhere diagonal and far enough
    viewer.cam.lookat[:] = center
    viewer.cam.distance  = max(2.0, 2.0 * extent)
    viewer.cam.azimuth   = 135
    viewer.cam.elevation = -25

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
