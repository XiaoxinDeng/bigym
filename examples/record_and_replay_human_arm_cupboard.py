import imageio
import numpy as np
from tqdm import tqdm, trange
from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards_with_human_arm import _HumanArmCupboardsInteractionEnv, HumanArmCupboardsOpenAll
from bigym.envs.cupboards import CupboardsOpenAll
from demonstrations.utils import Metadata
from demonstrations.demo_store import DemoStore
from demonstrations.demo_player import DemoPlayer
from demonstrations.demo import Demo
from demonstrations.demo_converter import DemoConverter
from bigym.const import CACHE_PATH
from bigym.action_modes import PelvisDof
import os
import mujoco
from mojo.elements import Geom
from bigym.utils.physics_utils import has_collided_collections


# Get demonstrations from DemoStore In case users do not have demos installed
demo_store = DemoStore()
demos = demo_store.pull_demos()
# Load demo from CACHE
filedir = f"{CACHE_PATH}/demonstrations/0.9.0/CupboardsOpenAll/"
filedir2 = 'JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/'
files = os.listdir(os.path.join(filedir, filedir2))
filename = os.path.join(filedir, filedir2, files[0])
assert os.path.exists(filename), f"Demo path invalid: {filename} "
demo = Demo.from_safetensors(filename)

# Set variables
n_steps = 3000
render = True
writer = imageio.get_writer("human_cupboard_demo.mp4", fps=30)
control_frequency = 50

fps = 30
frame_dt = 1.0 / fps
sim_t = 0.0
next_frame_t = 0.0
n_timesteps = len(demo.timesteps)
if n_steps is None or n_steps < 1:
    n_steps = n_timesteps
else:
    n_steps = min(n_timesteps, n_steps) # ensure steps in bound

def make_state_buffer(physics):
    m = physics.model.ptr
    n = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return np.zeros(n, dtype=np.float64)


def _snapshot(physics, buf):
    m = physics.model.ptr
    d = physics.data.ptr
    mujoco.mj_getState(m, d, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)

def _restore(physics, buf):
    m = physics.model.ptr
    d = physics.data.ptr
    mujoco.mj_setState(m, d, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(m, d)

def copy_state(env_src, env_dst, buf):
    # MuJoCo arrays
    _snapshot(env_src.mojo.physics, buf)
    _restore(env_dst.mojo.physics, buf)

    # Human script state
    hs = env_src.humanarms[0]
    hd = env_dst.humanarms[0]
    hd._CURRENT_TIME = hs._CURRENT_TIME
    hd._qpos_target[:] = hs._qpos_target
    hd._ctrl_target[:] = hs._ctrl_target

    # Floating-base controller buffers (BiGym-specific)
    fbs = env_src.robot.floating_base
    fbd = env_dst.robot.floating_base
    if fbs is not None and fbd is not None:
        fbd._accumulated_actions[:] = fbs._accumulated_actions
        fbd._last_action[:] = fbs._last_action


def _robot_colliders(env):
    geoms = []
    for geom_mjcf in env.robot._body.mjcf.find_all("geom"):
        g = Geom(env.mojo, geom_mjcf)
        if g.is_collidable():
            geoms.append(g)
    return geoms

def _has_penetration(physics, colliders_1, colliders_2, pen_eps=0.0):
    ids_1 = set(physics.bind([c.mjcf for c in colliders_1]).element_id)
    ids_2 = set(physics.bind([c.mjcf for c in colliders_2]).element_id)
    for c in physics.data.contact:
        if c.dist > pen_eps:
            continue
        if ((c.geom1 in ids_1 and c.geom2 in ids_2) or
            (c.geom2 in ids_1 and c.geom1 in ids_2)):
            return True
    return False


def _has_collision(physics, colliders_1, colliders_2, margin=0.001):
    if margin is None:
        return has_collided_collections(physics, colliders_1, colliders_2)
    ids_1 = set(physics.bind([c.mjcf for c in colliders_1]).element_id)
    ids_2 = set(physics.bind([c.mjcf for c in colliders_2]).element_id)
    for contact in physics.data.contact:
        if contact.dist > margin:
            continue
        if (contact.geom1 in ids_1 and contact.geom2 in ids_2) or (
            contact.geom2 in ids_1 and contact.geom1 in ids_2
        ):
            return True
    return False

def freeze_human_if_contact(env, robot_colliders):
    human = env.humanarms[0]
    if _has_penetration(env.mojo.physics, human.colliders, robot_colliders, pen_eps=0.0):
        human._CURRENT_TIME -= env.get_dt()
        return True
    return False

def geom_ids_from_colliders(physics, colliders):
    # element_id for geoms in dm_control bind
    ids = np.array(physics.bind([c.mjcf for c in colliders]).element_id, dtype=np.int32)
    return np.unique(ids)




def will_collide_within(env_pred, horizon_s, action, hit_thresh=0.01, step_dt=None):
    physics = env_pred.mojo.physics
    m = physics.model.ptr
    d = physics.data.ptr
    human = env_pred.humanarms[0]

    step_dt = float(step_dt or env_pred.get_dt())
    sub_steps = env_pred._sub_steps_count
    steps = int(np.ceil(horizon_s / step_dt))

    # early check
    dist0 = min_geom_distance(m, d, human_geom_ids_pred, robot_geom_ids_pred, distmax=hit_thresh)
    if dist0 < hit_thresh:
        return True, 0.0

    for i in range(steps):
        human._on_step(step_dt)
        mujoco.mj_forward(m, d)

        env_pred.action_mode.step(action)
        for _ in range(sub_steps - 1):
            env_pred.mojo.step()

        dist = min_geom_distance(m, d, human_geom_ids_pred, robot_geom_ids_pred, distmax=hit_thresh)
        if dist < hit_thresh:
            return True, (i + 1) * step_dt

    return False, None



def get_dof_ids(env):
    m = env.mojo.physics.model.ptr
    pelvis = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/pelvis_x"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/pelvis_y"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/pelvis_z"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/pelvis_rz"),
    ]
    pelvis_dofs = [int(m.jnt_dofadr[j]) for j in pelvis]

    fb_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/h1_floating_base/h1_floating_base")
    fb_dof = int(m.jnt_dofadr[fb_j])

    return pelvis_dofs, [fb_dof]

def set_resistance(env, dof_ids, damping=None, frictionloss=None):
    m = env.mojo.physics.model.ptr
    if damping is not None:
        m.dof_damping[dof_ids] = damping
    if frictionloss is not None:
        m.dof_frictionloss[dof_ids] = frictionloss
def min_contact_dist(physics, colliders_1, colliders_2):
    ids_1 = set(physics.bind([c.mjcf for c in colliders_1]).element_id)
    ids_2 = set(physics.bind([c.mjcf for c in colliders_2]).element_id)
    md = None
    for c in physics.data.contact:
        if ((c.geom1 in ids_1 and c.geom2 in ids_2) or
            (c.geom2 in ids_1 and c.geom1 in ids_2)):
            md = c.dist if md is None else min(md, c.dist)
    return md


env = HumanArmCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, 
                                        absolute=True, 
                                        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]),
    render_mode="rgb_array",
    arm_action_mode="scripted",
    control_frequency=50,
); env.reset()

env_pred = HumanArmCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, 
    absolute=True, 
    floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]),
    render_mode="rgb_array",
    arm_action_mode="scripted",
    control_frequency=50,); env_pred.reset()

def min_geom_distance(model, data, ids_a, ids_b, distmax=0.2):
    # distmax: early-exit threshold (m). Set to your gate/hit threshold.
    frompos = np.zeros(3, dtype=np.float64)
    topos   = np.zeros(3, dtype=np.float64)
    best = distmax
    for ga in ids_a:
        for gb in ids_b:
            # MuJoCo C API: mj_geomDistance(m,d,ga,gb,distmax,frompos,topos) -> distance
            dist = mujoco.mj_geomDistance(model, data, int(ga), int(gb), best, frompos, topos)
            if dist < best:
                best = float(dist)
                if best <= 0.0:   # penetration
                    return best
    return best

horizon_s = 0.6  # collision lookahead threshold in seconds
t = 0
paused = False
last_safe_action = demo.timesteps[0].executed_action.copy()
PRED_EVERY = 5  # 50Hz / 5 = 10Hz
buf = make_state_buffer(env.mojo.physics)
robot_colliders = _robot_colliders(env)
robot_colliders_pred = _robot_colliders(env_pred)
pbar = tqdm(total=n_steps, desc="Replaying demo", dynamic_ncols=True)
HIT_MARGIN = 0.1

human_geom_ids      = geom_ids_from_colliders(env.mojo.physics, env.humanarms[0].colliders)
robot_geom_ids      = geom_ids_from_colliders(env.mojo.physics, robot_colliders)
human_geom_ids_pred = geom_ids_from_colliders(env_pred.mojo.physics, env_pred.humanarms[0].colliders)
robot_geom_ids_pred = geom_ids_from_colliders(env_pred.mojo.physics, robot_colliders_pred)

HIT_THRESH = 0.01  # 1cm "about to touch" (tune down later)
NEAR_THRESH = 0.15  # 15cm gate (start generous)
m = env.mojo.physics.model.ptr
d = env.mojo.physics.data.ptr
near_dist = min_geom_distance(m, d, human_geom_ids, robot_geom_ids, distmax=NEAR_THRESH)
near = near_dist < NEAR_THRESH

# Make human arm geoms non-collidable with robot
# human = env.humanarms[0]
# for g in human.colliders:
#     mj = g.mjcf
#     mj.contype = 0
#     mj.conaffinity = 0

while t < n_steps:
    timestep = demo.timesteps[t]
    proposed = timestep.executed_action.copy()
    will_hit = False
    ttc = None

    # test collision on the proposed action
    if t % PRED_EVERY == 0 or paused:
        # near = _has_collision(env.mojo.physics, env.humanarms[0].colliders, robot_colliders, margin=0.10)  # 2cm proximity
        # if near:
        #     tqdm.write(f"[PRED] running lookahead at t={t}")
        #     # only then run expensive lookahead
        copy_state(env, env_pred, buf)
        will_hit, ttc = will_collide_within(env_pred, horizon_s, proposed, hit_thresh=0.01)

        # else:
        #     will_hit = False
            # tqdm.write(f"[PRED] gate blocked at t={t}")

    if will_hit:
        paused = True
        tqdm.write(f"[PAUSE] t={t}, ttc={ttc}")
    else:
        paused = False
        last_safe_action = proposed  # only update when safe

    # If paused, KEEP executing last_safe_action (do not zero base)
    action = last_safe_action if paused else proposed

    obs, reward, termination, truncation, info = env.step(action)

    if paused:
        # freeze human if actually penetrating (keep pen-based check!)
        freeze_human_if_contact(env, robot_colliders)

        # stay on the same demo timestep until safe again
        continue

    # only advance demo time when not paused
    t += 1
    pbar.update(1)   # update only when timestep advances

    # Optional: show extra info
    pbar.set_postfix(paused=paused, ttc=ttc)

    # Write frame in simulation time
    sim_t += env.get_dt()  # env_dt = opt.timestep * substeps
    if sim_t >= next_frame_t:
        frame = env.render()
        if frame is None: raise RuntimeError("env.render() returned None; use direct MuJoCo rendering.")
        writer.append_data(frame)
        next_frame_t += frame_dt

writer.close()
env.close()

