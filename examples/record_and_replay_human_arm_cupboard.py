from posixpath import sep

import imageio
import numpy as np
from tqdm import tqdm, trange
from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards_with_human_arm import HumanArmCupboardsOpenAll
from demonstrations.demo_store import DemoStore
from demonstrations.demo import Demo
from bigym.const import CACHE_PATH
from bigym.action_modes import PelvisDof
import os
import mujoco
from demo_utils import *
from demo_utils import GeomHighlighter
from demonstrations.demo_recorder import DemoRecorder
from bigym.utils.observation_config import ObservationConfig
from bigym.utils.observation_config import ObservationConfig, CameraConfig

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
save_dir = os.path.join("HumanArm", "HumanArmCupboardsOpenAll", filedir2)
os.makedirs(save_dir, exist_ok=True)

# Set variables
n_steps = None
render = True
write_demo_video = False
if write_demo_video:
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

observation_config = ObservationConfig(
    cameras=[
            CameraConfig(
                name="right_wrist",
                rgb=True,
                depth=False,
                resolution=(128, 128),
            ),
            CameraConfig(
                name="left_wrist",
                rgb=True,
                depth=False,
                resolution=(128, 128),
            ),
            CameraConfig(
                name="head",
                rgb=True,
                depth=False,
                resolution=(128, 128),
            ),
            CameraConfig(
                name="external",
                rgb=True,
                depth=False,
                resolution=(128, 128),
            ),
       ],
    proprioception=True,
    privileged_information=True,
)

env = HumanArmCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, 
                                        absolute=True, 
                                        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]),
    observation_config=observation_config,
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

recorder = DemoRecorder(save_dir)
recorder.record(env, lightweight_demo=True)

phys_pred = env_pred.mojo.physics
m_pred = phys_pred.model.ptr
m = env.mojo.physics.model.ptr
d = env.mojo.physics.data.ptr


PAUSE_DIST = 0.01 # pause if within the distance (positive allowed)
RESUME_DIST = 0.06 # resume only if >6cm
COLLISION_MARGIN = 0.06  # must be >= RESUME_DIST; margin for contact-based pausing (meters, roughly human fingertip thickness)
MAX_PAUSE_STEPS = 500 # np.inf  # safety valve against infinite deadlock
RESUME_DWELL = 15   # must be safe for 15 checks (e.g. 15 * PRED_EVERY/50 sec)
RAMP_STEPS = 50         # e.g. 20 ramp steps @ 50Hz = 0.4s
PRED_EVERY = 5  # 50Hz / 5 = 10Hz

t = 0
demo_t = t
safe_count = 0
ramp_k = RAMP_STEPS      # "not ramping" initially
paused = False
pause_steps = 0
prev_paused = False
resume_from_action = None  # for smooth ramping when leaving pause
last_safe_action = demo.timesteps[0].executed_action.copy()
buf = make_state_buffer(env.mojo.physics)
pbar = tqdm(total=n_steps, initial=t, desc="Replaying demo", dynamic_ncols=True)

robot_ids_pred = collidable_ids_with_prefix(m_pred, "h1/")  # collidable robot geoms (exclude non-collidable markers)
human_ids_pred = get_arm_geo_ids(m_pred)  # human arm geoms
hl = GeomHighlighter(env.mojo.physics, visible_group=2, env=env, env_pred=env_pred)

set_margins_for_sets(env_pred.mojo.physics, robot_ids_pred, margin=COLLISION_MARGIN)
set_margins_for_sets(env_pred.mojo.physics, human_ids_pred, margin=COLLISION_MARGIN)

while t < n_steps:
    timestep = demo.timesteps[demo_t]
    proposed = timestep.executed_action.copy()
    will_hit = False
    ttc = None

    # >>> Collision check <<<
    if t % PRED_EVERY == 0 or paused:
        copy_state(env, env_pred, buf)
        _ = env_pred.step(proposed)
        
        # contact-based (stable when touching)
        c_hit, cdist, cg1, cg2 = pair_min_contact_dist_between_sets(
            phys_pred, human_ids_pred, robot_ids_pred, dist_max=COLLISION_MARGIN  # use margin value
        )
       
        # pause condition: either actually touching OR too close
        pause_now = c_hit or (cdist < PAUSE_DIST)

        # resume condition: no contact AND far enough (with dwell)
        resume_ok = (not c_hit) and (cdist > RESUME_DIST)

        if not paused and pause_now:
            paused = True
            safe_count = 0
            # choose highlight ids (prefer contact pair if available)
            if cg1 != -1:
                name1 = mujoco.mj_id2name(m_pred, mujoco.mjtObj.mjOBJ_GEOM, cg1) or ""
                name2 = mujoco.mj_id2name(m_pred, mujoco.mjtObj.mjOBJ_GEOM, cg2) or ""
            tqdm.write(f"[PAUSE] t={t} c_hit={c_hit} cdist={cdist} {name1} <-> {name2}")

        elif paused:
            if resume_ok:
                safe_count += 1
            else:
                safe_count = 0

            if safe_count >= RESUME_DWELL:
                paused = False
                safe_count = 0
                tqdm.write(f"[RESUME] t={t} cdist={cdist:.4f}")

    # >>> Choose action <<<
    # When paused, we could hold the last action
    if paused:
        
        # highlight the pair in the *rendered* env
        hl.highlight_pred_contact_pair(cg1, cg2, rgba=(1, 0, 0, 1), highlight_body_visual=True)     # red
        # action = last_safe_action.copy()
        action = make_pause_hold_action(env, last_sent_action=last_safe_action)   # <-- hard stop at current pose
        zero_floating_base_velocity(env)  

        # stay on the same demo timestep until safe again
        pause_steps += 1
        ramp_k = RAMP_STEPS  # reset ramp state
        resume_from_action = last_safe_action.copy()
        if pause_steps > MAX_PAUSE_STEPS:
            paused = False
            pause_steps = 0
            tqdm.write(f"[FORCE RESUME] t={t} after {MAX_PAUSE_STEPS} pause steps")
    # when leaving pause, do a smooth ramping back to the proposed action instead of an instant jump
    else:
        hl.clear()

        # If we JUST resumed this step, initialize the ramp ONCE
        if prev_paused:
            ramp_k = 0
            # IMPORTANT: ramp starts from the command we were holding during pause
            if resume_from_action is None:
                resume_from_action = last_safe_action.copy()

        # Decide what action to send (ramp or direct)
        if ramp_k < RAMP_STEPS:
            alpha = (ramp_k + 1) / RAMP_STEPS
            action = (1 - alpha) * resume_from_action + alpha * proposed
            ramp_k += 1
        else:
            action = proposed.copy()
        
        # Advance demo index ONLY when not paused
        demo_t += 1
        last_safe_action = action.copy()   # <-- store what you ACTUALLY sent, not proposed
        pause_steps = 0

    
    output_timestep = env.step(action)
    recorder.add_timestep(output_timestep, action)

    # only advance demo time when not paused
    t += 1
    prev_paused = paused
    
    pbar.update(1)   # update only when timestep advances

    # Optional: show extra info
    pbar.set_postfix(paused=paused, ttc=ttc)

    # Write frame in simulation time
    sim_t += env.get_dt()  # env_dt = opt.timestep * substeps
    if sim_t >= next_frame_t and write_demo_video:
        frame = env.render()
        if frame is None: raise RuntimeError("env.render() returned None; use direct MuJoCo rendering.")
        writer.append_data(frame)
        next_frame_t += frame_dt

recorder.save_demo()
recorder.stop()
if write_demo_video:
    writer.close()
env.close()

