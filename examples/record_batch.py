import os
import json
import imageio
import numpy as np
import mujoco

from tqdm import tqdm
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.envs.cupboards_with_human_arm import HumanArmDrawerTopOpen
from demonstrations.demo_store import DemoStore
from demonstrations.demo import Demo
from demonstrations.demo_recorder import DemoRecorder
from bigym.const import CACHE_PATH
from bigym.utils.observation_config import ObservationConfig, CameraConfig

from demo_utils import (
    disable_arm_collisions,
    read_manifest_json,
    get_successful_demo_paths,
    make_state_buffer,
    copy_state,
    clamp_action,
    safe_step_pred,
    pair_min_contact_dist_between_sets,
    collidable_ids_with_prefix,
    get_arm_geo_ids,
    GeomHighlighter,
    build_action_joint_mapping_from_ranges,
    set_margins_for_sets,
    make_pause_hold_action_hybrid,
    zero_floating_base_velocity,
    summarize_manifest
)


CLASS_NAME = "DrawerTopOpen"

LABEL_MOVE = 0
LABEL_PAUSE = 1
LABEL_RESUME = 2

# -------------------------
# Paths
# -------------------------
root_dir = f"{CACHE_PATH}/demonstrations/0.9.0/"
joint_dir = "JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/"
target_dir = os.path.join(root_dir, CLASS_NAME, joint_dir)
data_save_dir = os.path.join(root_dir, f"HumanArm{CLASS_NAME}", joint_dir)
video_save_dir = "demo_videos"
# label_save_dir = os.path.join(data_save_dir, "mode_labels ")
# save_mode_labels = False
result_manifest_path = os.path.join(data_save_dir, "batch_result_manifest.json")

os.makedirs(data_save_dir, exist_ok=True)
os.makedirs(video_save_dir, exist_ok=True)
# os.makedirs(label_save_dir, exist_ok=True)

# -------------------------
# Replay settings
# -------------------------
n_demo_steps = None
write_demo_video = True
save_demo_to_disk = True
# save_raw_mode_labels  = True
control_frequency = 50
fps = 30
frame_dt = 1.0 / fps

PAUSE_DIST = 0.08
RESUME_DIST = 0.10
COLLISION_MARGIN = 0.10
MAX_PAUSE_STEPS = np.inf
RESUME_DWELL = 15
RAMP_STEPS = 50
PRED_EVERY_FAR = 3
PRED_EVERY_NEAR = 1
LOOKAHEAD_H = 5

PRINT_DEBUG_MSG = False

# -------------------------
# Observation config
# -------------------------
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

def collapse_raw_labels_to_demo_steps(raw_mode_labels, raw_demo_indices, replay_steps):
    per_step_labels = np.full(replay_steps, LABEL_MOVE, dtype=np.int64)

    # priority: PAUSE > RESUME > MOVE
    for step_idx in range(replay_steps):
        labs = [
            lab for lab, idx in zip(raw_mode_labels, raw_demo_indices)
            if idx == step_idx
        ]
        if len(labs) == 0:
            per_step_labels[step_idx] = LABEL_MOVE
        elif LABEL_PAUSE in labs:
            per_step_labels[step_idx] = LABEL_PAUSE
        elif LABEL_RESUME in labs:
            per_step_labels[step_idx] = LABEL_RESUME
        else:
            per_step_labels[step_idx] = LABEL_MOVE

    return per_step_labels

def label_to_name(label: int) -> str:
    if label == LABEL_MOVE:
        return "MOVE"
    if label == LABEL_PAUSE:
        return "PAUSE"
    if label == LABEL_RESUME:
        return "RESUME"
    return "UNKNOWN"


def make_env():
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
    return env


def make_pred_env():
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
    return env_pred


def run_one_demo(filename, env, env_pred, demo_index=0, num_demos=1):
    assert os.path.exists(filename), f"Demo path invalid: {filename}"
    demo = Demo.from_safetensors(filename)

    # -------------------------
    # Per-demo reset
    # -------------------------
    env.reset()
    env_pred.reset()
    disable_arm_collisions(env.mojo.physics)

    sim_t = 0.0
    next_frame_t = 0.0
    step_t = 0
    demo_t = 0
    safe_count = 0
    ramp_k = RAMP_STEPS
    paused = False
    pause_steps = 0
    prev_paused = False
    resume_from_action = None
    success = False
    ttc = None
    cdist = float("nan")
    cg1, cg2 = -1, -1

    n_timesteps = len(demo.timesteps)
    if n_demo_steps is None or n_demo_steps < 1:
        replay_steps = n_timesteps
    else:
        replay_steps = min(n_timesteps, n_demo_steps)

    last_safe_action = demo.timesteps[0].executed_action.copy()

    # -------------------------
    # Writer / recorder
    # -------------------------
    video_filename = os.path.join(
        video_save_dir, f"human_cupboard_{CLASS_NAME}_demo_{demo.uuid}.mp4"
    )

    writer = imageio.get_writer(video_filename, fps=fps) if write_demo_video else None

    recorder = DemoRecorder(data_save_dir)
    if save_demo_to_disk:
        recorder.record(env, lightweight_demo=True)
        target_demo_uuid = str(recorder.demo.uuid)
    else:
        target_demo_uuid = str(demo.uuid)

    target_demo_path = None

    # -------------------------
    # Prediction helpers
    # -------------------------
    phys_pred = env_pred.mojo.physics
    m_pred = phys_pred.model.ptr

    robot_ids_pred = collidable_ids_with_prefix(m_pred, "h1/")
    human_ids_pred = get_arm_geo_ids(m_pred)

    set_margins_for_sets(env_pred.mojo.physics, robot_ids_pred, margin=COLLISION_MARGIN)
    set_margins_for_sets(env_pred.mojo.physics, human_ids_pred, margin=COLLISION_MARGIN)

    hl = GeomHighlighter(env.mojo.physics, visible_group=2, env=env, env_pred=env_pred)
    action_joint_jids = build_action_joint_mapping_from_ranges(
        env, prefix="h1/", start_dim=4, end_dim=14
    )
    buf = make_state_buffer(env.mojo.physics)

    tqdm.write(f"\n=== Replaying demo {demo.uuid} ===")
    tqdm.write(f"Source: {filename}")
    tqdm.write(f"action_dim: {env.action_space.shape[0]}")
    tqdm.write(f"action_mode type: {type(env.action_mode)}")
    tqdm.write(f"Save video to: {video_filename}")

    pbar = tqdm(
        total=replay_steps,
        initial=demo_t,
        desc=f"Demo {demo_index + 1}/{num_demos}",
        position=1,
        leave=False,
        dynamic_ncols=True,
    )

    # -------------------------
    # Raw per-env-step traces
    # These are aligned with the RECORDED target demo timesteps.
    # -------------------------
    raw_mode_labels = []
    raw_mode_names = []
    raw_pause_flags = []
    raw_demo_indices = []
    raw_proposed_actions = []
    raw_executed_actions = []
    raw_success_flags = []

    exception_msg = None

    try:
        while demo_t < replay_steps:
            timestep = demo.timesteps[demo_t]
            proposed = timestep.executed_action.copy()
            ttc = None

            # Source-demo index currently being replayed.
            # Kept only for debugging / provenance.
            label_demo_idx = demo_t

            # ---------- Collision check / pause-resume logic ----------
            pred_every = PRED_EVERY_NEAR if paused else PRED_EVERY_FAR
            if (step_t % pred_every == 0) or paused:
                copy_state(env, env_pred, buf)

                if paused:
                    pred_action_now = make_pause_hold_action_hybrid(
                        env_pred, action_joint_jids, last_safe_action
                    )
                    pred_action_now = clamp_action(env_pred, pred_action_now)
                    _, pred_ok, _ = safe_step_pred(env_pred, pred_action_now)
                    if pred_ok:
                        zero_floating_base_velocity(env_pred)
                else:
                    proposed = clamp_action(env_pred, proposed)
                    _, pred_ok, _ = safe_step_pred(env_pred, proposed)

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
                        name1 = mujoco.mj_id2name(
                            m_pred, mujoco.mjtObj.mjOBJ_GEOM, int(cg1)
                        ) or ""
                        name2 = mujoco.mj_id2name(
                            m_pred, mujoco.mjtObj.mjOBJ_GEOM, int(cg2)
                        ) or ""

                    if PRINT_DEBUG_MSG:
                        tqdm.write(
                            f"[PAUSE] step_t={step_t} c_hit={c_hit} "
                            f"cdist={cdist:.4f} {name1} <-> {name2}"
                        )

                elif paused:
                    safe_count = safe_count + 1 if resume_clear else 0

                    if safe_count >= RESUME_DWELL:
                        ok = True
                        copy_state(env, env_pred, buf)

                        for _ in range(LOOKAHEAD_H):
                            proposed = clamp_action(env_pred, proposed)
                            _, pred_ok2, _ = safe_step_pred(env_pred, proposed)

                            if not pred_ok2:
                                ok = False
                                cg1, cg2 = -1, -1
                                break

                            c_hit2, cdist2, cg1_2, cg2_2 = pair_min_contact_dist_between_sets(
                                phys_pred,
                                human_ids_pred,
                                robot_ids_pred,
                                dist_max=COLLISION_MARGIN,
                            )
                            if c_hit2 or (cdist2 < PAUSE_DIST):
                                ok = False
                                cg1, cg2 = cg1_2, cg2_2
                                break

                        if ok:
                            paused = False
                            safe_count = 0
                            pause_steps = 0
                            if PRINT_DEBUG_MSG:
                                tqdm.write(f"[RESUME] step_t={step_t} cdist={cdist:.4f}")
                        else:
                            safe_count = 0

            # ---------- Choose real action + mode label ----------
            if paused:
                current_mode = LABEL_PAUSE

                if cg1 != -1:
                    hl.highlight_pred_contact_pair(
                        cg1, cg2, rgba=(1, 0, 0, 1), highlight_body_visual=True
                    )

                action = make_pause_hold_action_hybrid(
                    env, action_joint_jids, last_safe_action
                )
                zero_floating_base_velocity(env)
                last_safe_action = action.copy()
                pause_steps += 1
                ramp_k = RAMP_STEPS
                resume_from_action = last_safe_action.copy()

                if pause_steps > MAX_PAUSE_STEPS:
                    paused = False
                    safe_count = 0
                    pause_steps = 0
                    tqdm.write(
                        f"[FORCE RESUME] step_t={step_t} after {MAX_PAUSE_STEPS} pause steps"
                    )

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

            # ---------- Step real env ----------
            action = clamp_action(env, action)
            output_timestep = env.step(action)
            success = bool(env.success)

            # Attach aligned label to the timestep being saved.
            # This makes timestep[i] and mode_label[i] refer to the same executed transition.
            if output_timestep.info is None:
                output_timestep.info = {}

            output_timestep.info["mode_label"] = int(current_mode)
            output_timestep.info["mode_name"] = label_to_name(current_mode)
            output_timestep.info["paused"] = bool(paused)
            output_timestep.info["source_demo_idx"] = int(min(label_demo_idx, replay_steps - 1))

            # ---------- Record raw per-env-step traces ----------
            raw_mode_labels.append(int(current_mode))
            raw_mode_names.append(label_to_name(current_mode))
            raw_pause_flags.append(bool(paused))
            raw_demo_indices.append(int(min(label_demo_idx, replay_steps - 1)))
            raw_proposed_actions.append(proposed.copy())
            raw_executed_actions.append(action.copy())
            raw_success_flags.append(success)

            if save_demo_to_disk:
                recorder.add_timestep(output_timestep, action)

            arm_dbg = env.humanarms[0].get_debug_keepout_state()
            if PRINT_DEBUG_MSG and (step_t % 10 == 0) and (arm_dbg["active"] or paused):
                tqdm.write(
                    f"[DBG] step_t={step_t} paused={paused} mode={label_to_name(current_mode)} "
                    f"zone={arm_dbg['zone']} clear={arm_dbg['clear']:.4f} "
                    f"active={arm_dbg['active']} push={arm_dbg['push']:.4f} "
                    f"nxy=({arm_dbg['nxy'][0]:.3f},{arm_dbg['nxy'][1]:.3f}) "
                    f"shadow_cdist={cdist:.4f}"
                )

            step_t += 1
            prev_paused = paused
            pbar.set_postfix(
                paused=paused,
                mode=label_to_name(current_mode),
                success=success,
                ttc=ttc,
            )

            sim_t += env.get_dt()
            if sim_t >= next_frame_t and write_demo_video:
                frame = env.render()
                if frame is None:
                    raise RuntimeError("env.render() returned None; use direct MuJoCo rendering.")
                writer.append_data(frame)
                next_frame_t += frame_dt

    except Exception as e:
        exception_msg = repr(e)
        tqdm.write(f"[ERROR] demo {demo.uuid} failed with exception: {exception_msg}")

    finally:
        pbar.close()
        if save_demo_to_disk:
            target_demo_path = recorder.save_demo()
            recorder.stop()
        if writer is not None:
            writer.close()

    final_drawer_state = env.cabinet_drawers.get_state()[-1]
    tqdm.write(f"success: {str(success)} | Final Drawer State: {final_drawer_state}\n")

    # -------------------------
    # Debug-only collapsed labels in source-demo index space
    # Not for training alignment.
    # -------------------------
    collapsed_mode_labels = collapse_raw_labels_to_demo_steps(
        raw_mode_labels=raw_mode_labels,
        raw_demo_indices=raw_demo_indices,
        replay_steps=replay_steps,
    )


    return {
        "uuid": str(target_demo_uuid),
        "source_uuid": str(demo.uuid),
        "source_path": filename,
        "target_path": str(target_demo_path) if target_demo_path is not None else None,
        "video_path": video_filename if write_demo_video else None,
        "success": int(success),
        "final_drawer_state": float(final_drawer_state),
        "replay_steps": int(replay_steps),
        "executed_env_steps": int(step_t),
        "num_mode_labels": int(len(raw_mode_labels)),
        "num_collapsed_mode_labels": int(len(collapsed_mode_labels)),
        "exception": exception_msg,
    }
def main():
    demo_store = DemoStore()
    demo_store.pull_demos()

    manifest_path = os.path.join(target_dir, "manifest.json")
    manifest = read_manifest_json(manifest_path)
    filenames = get_successful_demo_paths(manifest)

    tqdm.write(f"Found {len(filenames)} successful demos in manifest")

    if len(filenames) == 0:
        raise RuntimeError(f"No successful demo paths found in {manifest_path}")

    env = make_env()
    env_pred = make_pred_env()

    results = []
    outer_pbar = tqdm(
        total=len(filenames),
        desc="Batch demos",
        position=0,
        dynamic_ncols=True,
    )

    try:
        for i, filename in enumerate(filenames):
            outer_pbar.set_postfix_str(os.path.basename(filename))
            tqdm.write(f"\n[{i + 1}/{len(filenames)}] Processing: {filename}")
            result = run_one_demo(filename, env, env_pred, demo_index=i, num_demos=len(filenames))
            results.append(result)

            with open(result_manifest_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            outer_pbar.update(1)

        outer_pbar.close()

    finally:
        env.close()
        env_pred.close()

    tqdm.write(f"\nSaved batch result manifest to: {result_manifest_path}")


if __name__ == "__main__":
    main()