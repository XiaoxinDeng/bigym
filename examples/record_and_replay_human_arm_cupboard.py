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
n_steps = 3000
render = True
write_demo_video = True
save_demo_to_disk = False
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
disable_arm_collisions(env.mojo.physics)

env_pred = HumanArmCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, 
                                        absolute=True, 
                                        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]),
    render_mode="rgb_array",
    arm_action_mode="scripted",
    control_frequency=50,); env_pred.reset()

recorder = DemoRecorder(save_dir)
if save_demo_to_disk:
    recorder.record(env, lightweight_demo=True)

phys_pred = env_pred.mojo.physics
m_pred = phys_pred.model.ptr
m = env.mojo.physics.model.ptr
d = env.mojo.physics.data.ptr


PAUSE_DIST = 0.05 # pause if within the distance (positive allowed)
RESUME_DIST = 0.07 # resume only if >6cm
COLLISION_MARGIN = 0.08  # must be >= RESUME_DIST; margin for contact-based pausing (meters, roughly human fingertip thickness)
MAX_PAUSE_STEPS = np.inf # np.inf  # safety valve against infinite deadlock
RESUME_DWELL = 15   # must be safe for 15 checks (e.g. 15 * PRED_EVERY/50 sec)
RAMP_STEPS = 50         # e.g. 20 ramp steps @ 50Hz = 0.4s
PRED_EVERY = 5  # 50Hz / 5 = 10Hz
PRED_EVERY_FAR = 5
PRED_EVERY_NEAR = 1
NEAR_DIST = 0.10  # if within 10 cm, check every step

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
action_joint_jids = build_action_joint_mapping_from_ranges(env, prefix="h1/", start_dim=4, end_dim=14)

set_margins_for_sets(env_pred.mojo.physics, robot_ids_pred, margin=COLLISION_MARGIN)
set_margins_for_sets(env_pred.mojo.physics, human_ids_pred, margin=COLLISION_MARGIN)


# -------------------------
# Fixed replay while-loop
# - While paused: shadow env follows the SAME "hold" control as real env (prevents shadow penetration/instability)
# - Resume decision uses:
#   (A) clearance in shadow under HOLD (matches reality during pause)
#   (B) optional short lookahead with PROPOSED only when about to resume
# -------------------------
print("action_dim:", env.action_space.shape[0])
print("action_mode type:", type(env.action_mode))

LOOKAHEAD_H = 5  # steps of lookahead with proposed when considering resume (e.g., 5 @ 50Hz = 0.1s)

while t < n_steps:
    timestep = demo.timesteps[demo_t]
    proposed = timestep.executed_action.copy()
    will_hit = False
    ttc = None

    # ---------- Collision check / pause-resume logic ----------
    pred_every = PRED_EVERY_NEAR if paused else PRED_EVERY_FAR
    if (t % pred_every == 0) or paused:
        # Keep env_pred synchronized with env
        copy_state(env, env_pred, buf)

        # 1) Step env_pred with the SAME action that env will take right now
        if paused:
            pred_action_now = make_pause_hold_action_hybrid(env_pred, action_joint_jids, last_safe_action)
            pred_action_now = clamp_action(env_pred, pred_action_now)
            _ = env_pred.step(pred_action_now)
            zero_floating_base_velocity(env_pred)
        else:
            proposed = clamp_action(env_pred, proposed)
            _ = env_pred.step(proposed)

        # 2) Compute contact/dist in env_pred after that step
        c_hit, cdist, cg1, cg2 = pair_min_contact_dist_between_sets(
            phys_pred, human_ids_pred, robot_ids_pred, dist_max=COLLISION_MARGIN
        )

        pause_now = c_hit or (cdist < PAUSE_DIST)
        resume_clear = (not c_hit) and (cdist > RESUME_DIST)

        # Transition into pause
        if not paused and pause_now:
            paused = True
            safe_count = 0
            ramp_k = RAMP_STEPS
            resume_from_action = last_safe_action.copy()
            pause_steps = 0

            name1 = name2 = ""
            if cg1 != -1:
                name1 = mujoco.mj_id2name(m_pred, mujoco.mjtObj.mjOBJ_GEOM, int(cg1)) or ""
                name2 = mujoco.mj_id2name(m_pred, mujoco.mjtObj.mjOBJ_GEOM, int(cg2)) or ""
            tqdm.write(f"[PAUSE] t={t} c_hit={c_hit} cdist={cdist:.4f} {name1} <-> {name2}")

        # Stay paused: dwell-based clearance, plus OPTIONAL lookahead before resuming
        elif paused:
            safe_count = safe_count + 1 if resume_clear else 0

            # Only consider resuming once we've been clear for long enough
            if safe_count >= RESUME_DWELL:
                # 3) Optional lookahead: if we resume, will PROPOSED collide soon?
                ok = True
                copy_state(env, env_pred, buf)  # reset shadow back to current real state

                # simulate LOOKAHEAD_H steps with proposed (or ramp-start) to test immediate collision risk
                for _ in range(LOOKAHEAD_H):
                    proposed = clamp_action(env_pred, proposed)
                    _ = env_pred.step(proposed)
                    c_hit2, cdist2, cg1_2, cg2_2 = pair_min_contact_dist_between_sets(
                        phys_pred, human_ids_pred, robot_ids_pred, dist_max=COLLISION_MARGIN
                    )
                    if c_hit2 or (cdist2 < PAUSE_DIST):
                        ok = False
                        cg1, cg2 = cg1_2, cg2_2  # for highlighting
                        break

                if ok:
                    paused = False
                    safe_count = 0
                    pause_steps = 0
                    tqdm.write(f"[RESUME] t={t} cdist={cdist:.4f}")
                else:
                    # remain paused; reset dwell so we require sustained clearance again
                    safe_count = 0

    # ---------- Choose action to send to the real env ----------
    if paused:
        # Highlight predicted contact pair (if any)
        if cg1 != -1:
            hl.highlight_pred_contact_pair(cg1, cg2, rgba=(1, 0, 0, 1), highlight_body_visual=True)

        # Hard stop at current pose (robot yields; human keeps moving)
        # TRUE HOLD for hybrid semantics (prevents controller fighting)
        action = make_pause_hold_action_hybrid(env, action_joint_jids, last_safe_action)
        zero_floating_base_velocity(env)
        last_safe_action = action.copy()   # <-- add this
        pause_steps += 1
        ramp_k = RAMP_STEPS
        resume_from_action = last_safe_action.copy()

        if pause_steps > MAX_PAUSE_STEPS:
            paused = False
            safe_count = 0
            pause_steps = 0
            tqdm.write(f"[FORCE RESUME] t={t} after {MAX_PAUSE_STEPS} pause steps")

    else:
        hl.clear()

        # If we JUST resumed, initialize ramp ONCE
        if prev_paused:
            ramp_k = 0
            if resume_from_action is None:
                resume_from_action = last_safe_action.copy()

        # Ramp back to proposed to avoid a jump
        if ramp_k < RAMP_STEPS:
            alpha = (ramp_k + 1) / RAMP_STEPS
            action = (1 - alpha) * resume_from_action + alpha * proposed
            ramp_k += 1
        else:
            action = proposed.copy()

        # Advance demo index ONLY when not paused
        demo_t += 1
        last_safe_action = action.copy()
        pause_steps = 0

    # ---------- Step real env ----------
    action = clamp_action(env, action)
    output_timestep = env.step(action)
    if save_demo_to_disk:
        recorder.add_timestep(output_timestep, action)

    # Always advance time step counter t (your design: arm moves each env.step)
    t += 1
    prev_paused = paused

    pbar.update(1)
    pbar.set_postfix(paused=paused, ttc=ttc)

    sim_t += env.get_dt()
    if sim_t >= next_frame_t and write_demo_video:
        frame = env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None; use direct MuJoCo rendering.")
        writer.append_data(frame)
        next_frame_t += frame_dt

if save_demo_to_disk:
    recorder.save_demo()
    recorder.stop()
if write_demo_video:
    writer.close()
env.close()

