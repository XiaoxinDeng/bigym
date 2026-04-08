from posixpath import sep

import imageio
import json
import numpy as np
from tqdm import tqdm, trange
from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards_with_human_arm import HumanArmDrawerTopOpen
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

# -------------------------
# Config
# -------------------------
CLASS_NAME = "DrawerTopOpen"
LABEL_MOVE = 0
LABEL_PAUSE = 1
LABEL_RESUME = 2

# Get demonstrations from DemoStore in case users do not have demos installed
demo_store = DemoStore()
demos = demo_store.pull_demos()

# Load demo from CACHE
root_dir = f"{CACHE_PATH}/demonstrations/0.9.0/"
Joint_dir = "JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/"
target_dir = os.path.join(root_dir, CLASS_NAME, Joint_dir)
data_save_dir = os.path.join(root_dir, f"HumanArm{CLASS_NAME}", Joint_dir)
video_save_dir = "demo_videos"
label_save_dir = os.path.join(data_save_dir, "mode_labels")

os.makedirs(data_save_dir, exist_ok=True)
os.makedirs(video_save_dir, exist_ok=True)
os.makedirs(label_save_dir, exist_ok=True)

files = os.listdir(target_dir)
manifest = read_manifest_json(os.path.join(target_dir, "manifest.json"))
filenames = get_successful_demo_paths(manifest)
filename = filenames[0]
assert os.path.exists(filename), f"Demo path invalid: {filename}"
demo = Demo.from_safetensors(filename)

# -------------------------
# Replay settings
# -------------------------
n_demo_steps = None
render = True
write_demo_video = True
save_demo_to_disk = False
save_mode_labels = True
control_frequency = 50
fps = 30
frame_dt = 1.0 / fps
sim_t = 0.0
next_frame_t = 0.0

n_timesteps = len(demo.timesteps)
if n_demo_steps is None or n_demo_steps < 1:
    n_demo_steps = n_timesteps
else:
    n_demo_steps = min(n_timesteps, n_demo_steps)

video_filename = f"{video_save_dir}/human_cupboard_{CLASS_NAME}_demo_{demo.uuid}.mp4"
label_filename = os.path.join(label_save_dir, f"{demo.uuid}_mode_labels.npz")
meta_filename = os.path.join(label_save_dir, f"{demo.uuid}_mode_labels.json")

print(f"Save video to: {video_filename}")
print(f"Save labels to: {label_filename}")

if write_demo_video:
    writer = imageio.get_writer(video_filename, fps=fps)

recorder = DemoRecorder(data_save_dir)

observation_config = ObservationConfig(
    cameras=[
        CameraConfig(name="right_wrist", rgb=True, depth=False, resolution=(128, 128)),
        CameraConfig(name="left_wrist", rgb=True, depth=False, resolution=(128, 128)),
        CameraConfig(name="head", rgb=True, depth=False, resolution=(128, 128)),
        CameraConfig(name="external", rgb=True, depth=False, resolution=(128, 128)),
    ],
    proprioception=True,
    privileged_information=True,
)

env = HumanArmDrawerTopOpen(
    action_mode=JointPositionActionMode(
        floating_base=True,
        absolute=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    ),
    observation_config=observation_config,
    render_mode="rgb_array",
    arm_action_mode="scripted",
    control_frequency=control_frequency,
)
env.reset()
disable_arm_collisions(env.mojo.physics)

env_pred = HumanArmDrawerTopOpen(
    action_mode=JointPositionActionMode(
        floating_base=True,
        absolute=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    ),
    render_mode="rgb_array",
    arm_action_mode="scripted",
    control_frequency=control_frequency,
)
env_pred.reset()

if save_demo_to_disk:
    recorder.record(env, lightweight_demo=True)

phys_pred = env_pred.mojo.physics
m_pred = phys_pred.model.ptr
m = env.mojo.physics.model.ptr
d = env.mojo.physics.data.ptr

PAUSE_DIST = 0.08
RESUME_DIST = 0.10
COLLISION_MARGIN = 0.10
MAX_PAUSE_STEPS = np.inf
RESUME_DWELL = 15
RAMP_STEPS = 50
PRED_EVERY_FAR = 3
PRED_EVERY_NEAR = 1
LOOKAHEAD_H = 5

step_t = 0
demo_t = 0
safe_count = 0
ramp_k = RAMP_STEPS
paused = False
pause_steps = 0
prev_paused = False
resume_from_action = None
last_safe_action = demo.timesteps[0].executed_action.copy()
buf = make_state_buffer(env.mojo.physics)

pbar = tqdm(total=n_demo_steps, initial=demo_t, desc="Replaying demo", dynamic_ncols=True)

robot_ids_pred = collidable_ids_with_prefix(m_pred, "h1/")
human_ids_pred = get_arm_geo_ids(m_pred)
hl = GeomHighlighter(env.mojo.physics, visible_group=2, env=env, env_pred=env_pred)
action_joint_jids = build_action_joint_mapping_from_ranges(env, prefix="h1/", start_dim=4, end_dim=14)

set_margins_for_sets(env_pred.mojo.physics, robot_ids_pred, margin=COLLISION_MARGIN)
set_margins_for_sets(env_pred.mojo.physics, human_ids_pred, margin=COLLISION_MARGIN)

print("action_dim:", env.action_space.shape[0])
print("action_mode type:", type(env.action_mode))

# -------------------------
# Mode label buffers
# One record per REAL env.step(...)
# -------------------------
mode_labels = []
mode_names = []
pause_flags = []
demo_indices = []
proposed_actions = []
executed_actions = []
success_flags = []

def label_to_name(label: int) -> str:
    if label == LABEL_MOVE:
        return "MOVE"
    if label == LABEL_PAUSE:
        return "PAUSE"
    if label == LABEL_RESUME:
        return "RESUME"
    return "UNKNOWN"

success = False

while demo_t < n_demo_steps:
    timestep = demo.timesteps[demo_t]
    proposed = timestep.executed_action.copy()
    ttc = None
    cdist = np.nan
    cg1 = -1
    cg2 = -1

    # ---------- Collision check / pause-resume logic ----------
    pred_every = PRED_EVERY_NEAR if paused else PRED_EVERY_FAR
    if (step_t % pred_every == 0) or paused:
        copy_state(env, env_pred, buf)

        if paused:
            pred_action_now = make_pause_hold_action_hybrid(env_pred, action_joint_jids, last_safe_action)
            pred_action_now = clamp_action(env_pred, pred_action_now)
            ts_pred, pred_ok, pred_reason = safe_step_pred(env_pred, pred_action_now)
            if pred_ok:
                zero_floating_base_velocity(env_pred)
        else:
            proposed = clamp_action(env_pred, proposed)
            ts_pred, pred_ok, pred_reason = safe_step_pred(env_pred, proposed)

        if not pred_ok:
            c_hit, cdist, cg1, cg2 = True, -np.inf, -1, -1
        else:
            c_hit, cdist, cg1, cg2 = pair_min_contact_dist_between_sets(
                phys_pred, human_ids_pred, robot_ids_pred, dist_max=COLLISION_MARGIN
            )

        pause_now = c_hit or (cdist < PAUSE_DIST)
        resume_clear = (not c_hit) and (cdist > RESUME_DIST)

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
            tqdm.write(f"[PAUSE] step_t={step_t} c_hit={c_hit} cdist={cdist:.4f} {name1} <-> {name2}")

        elif paused:
            safe_count = safe_count + 1 if resume_clear else 0

            if safe_count >= RESUME_DWELL:
                ok = True
                copy_state(env, env_pred, buf)

                for _ in range(LOOKAHEAD_H):
                    proposed = clamp_action(env_pred, proposed)
                    ts_pred, pred_ok2, pred_reason2 = safe_step_pred(env_pred, proposed)

                    if not pred_ok2:
                        ok = False
                        cg1, cg2 = -1, -1
                        break

                    c_hit2, cdist2, cg1_2, cg2_2 = pair_min_contact_dist_between_sets(
                        phys_pred, human_ids_pred, robot_ids_pred, dist_max=COLLISION_MARGIN
                    )
                    if c_hit2 or (cdist2 < PAUSE_DIST):
                        ok = False
                        cg1, cg2 = cg1_2, cg2_2
                        break

                if ok:
                    paused = False
                    safe_count = 0
                    pause_steps = 0
                    tqdm.write(f"[RESUME] step_t={step_t} cdist={cdist:.4f}")
                else:
                    safe_count = 0

    # ---------- Choose action + mode label ----------
    if paused:
        current_mode = LABEL_PAUSE

        if cg1 != -1:
            hl.highlight_pred_contact_pair(cg1, cg2, rgba=(1, 0, 0, 1), highlight_body_visual=True)

        action = make_pause_hold_action_hybrid(env, action_joint_jids, last_safe_action)
        zero_floating_base_velocity(env)
        last_safe_action = action.copy()
        pause_steps += 1
        ramp_k = RAMP_STEPS
        resume_from_action = last_safe_action.copy()

        if pause_steps > MAX_PAUSE_STEPS:
            paused = False
            safe_count = 0
            pause_steps = 0
            tqdm.write(f"[FORCE RESUME] step_t={step_t} after {MAX_PAUSE_STEPS} pause steps")

    else:
        hl.clear()

        if prev_paused:
            current_mode = LABEL_RESUME
            ramp_k = 0
            if resume_from_action is None:
                resume_from_action = last_safe_action.copy()
        else:
            current_mode = LABEL_MOVE

        if ramp_k < RAMP_STEPS:
            alpha = (ramp_k + 1) / RAMP_STEPS
            action = (1 - alpha) * resume_from_action + alpha * proposed
            ramp_k += 1
        else:
            action = proposed.copy()

        demo_t += 1
        last_safe_action = action.copy()
        pause_steps = 0
        pbar.update(1)

    # ---------- Record mode labels BEFORE step ----------
    mode_labels.append(current_mode)
    mode_names.append(label_to_name(current_mode))
    pause_flags.append(bool(paused))
    demo_indices.append(int(min(demo_t, n_demo_steps - 1)))
    proposed_actions.append(proposed.copy())
    executed_actions.append(action.copy())

    # ---------- Step real env ----------
    action = clamp_action(env, action)
    output_timestep = env.step(action)
    success = env.success
    success_flags.append(bool(success))

    if save_demo_to_disk:
        recorder.add_timestep(output_timestep, action)

    arm_dbg = env.humanarms[0].get_debug_keepout_state()
    if (step_t % 10 == 0) and (arm_dbg["active"] or paused):
        tqdm.write(
            f"[DBG] step_t={step_t} paused={paused} mode={label_to_name(current_mode)} "
            f"zone={arm_dbg['zone']} clear={arm_dbg['clear']:.4f} active={arm_dbg['active']} "
            f"push={arm_dbg['push']:.4f} "
            f"nxy=({arm_dbg['nxy'][0]:.3f},{arm_dbg['nxy'][1]:.3f}) "
            f"shadow_cdist={cdist:.4f}"
        )

    step_t += 1
    prev_paused = paused
    pbar.set_postfix(paused=paused, mode=label_to_name(current_mode), ttc=ttc)

    sim_t += env.get_dt()
    if sim_t >= next_frame_t and write_demo_video:
        frame = env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None; use direct MuJoCo rendering.")
        writer.append_data(frame)
        next_frame_t += frame_dt

pbar.close()

print(f"success: {success}")
print(env.cabinet_drawers.get_state()[-1])

# -------------------------
# Save sidecar labels
# -------------------------
if save_mode_labels:
    np.savez_compressed(
        label_filename,
        demo_uuid=str(demo.uuid),
        mode_labels=np.asarray(mode_labels, dtype=np.int64),
        pause_flags=np.asarray(pause_flags, dtype=np.bool_),
        demo_indices=np.asarray(demo_indices, dtype=np.int64),
        proposed_actions=np.asarray(proposed_actions, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        success_flags=np.asarray(success_flags, dtype=np.bool_),
        label_map=np.asarray(
            ["MOVE", "PAUSE", "RESUME"],
            dtype=object,
        ),
    )

    with open(meta_filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "demo_uuid": str(demo.uuid),
                "class_name": CLASS_NAME,
                "n_steps_recorded": len(mode_labels),
                "label_map": {
                    "0": "MOVE",
                    "1": "PAUSE",
                    "2": "RESUME",
                },
                "pause_dist": PAUSE_DIST,
                "resume_dist": RESUME_DIST,
                "resume_dwell": RESUME_DWELL,
                "ramp_steps": RAMP_STEPS,
                "control_frequency": control_frequency,
            },
            f,
            indent=2,
        )

if save_demo_to_disk:
    recorder.save_demo()
    recorder.stop()

if write_demo_video:
    writer.close()

env.close()
env_pred.close()