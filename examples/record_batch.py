from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import imageio
import numpy as np
from tqdm import tqdm

import mujoco

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.envs.cupboards_with_human_arm import HumanArmCupboardsOpenAll
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.const import CACHE_PATH

from demonstrations.demo_store import DemoStore
from demonstrations.demo import Demo
from demonstrations.demo_recorder import DemoRecorder
from collections import Counter

# your helpers
from demo_utils import (
    make_state_buffer,
    copy_state,
    collidable_ids_with_prefix,
    get_arm_geo_ids,
    set_margins_for_sets,
    pair_min_contact_dist_between_sets,
    GeomHighlighter,
    make_pause_hold_action,
    zero_floating_base_velocity,
)

# ----------------------------
# Config
# ----------------------------

@dataclass
class CameraSpec:
    name: str
    resolution: tuple[int, int] = (128, 128)
    rgb: bool = True
    depth: bool = False


@dataclass
class DemoRecordConfig:
    # I/O
    save_root: Path = Path("HumanArm") / "HumanArmCupboardsOpenAll"
    lightweight_demo: bool = True
    write_demo_video: bool = False
    video_fps: int = 30

    # Demo selection / limiting
    n_steps: Optional[int] = None  # None => full demo

    # Env/action
    control_frequency: int = 50
    floating_dofs: tuple[PelvisDof, ...] = (PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ)
    absolute: bool = True
    floating_base: bool = True
    arm_action_mode: str = "scripted"   # pass-through to env

    # Observations (only used for the main env; pred env can omit for speed)
    cameras: list[CameraSpec] = field(default_factory=lambda: [
        CameraSpec("right_wrist"),
        CameraSpec("left_wrist"),
        CameraSpec("head"),
        CameraSpec("external"),
    ])
    proprioception: bool = True
    privileged_information: bool = True

    # Collision pause/resume logic
    PAUSE_DIST: float = 0.01
    RESUME_DIST: float = 0.06
    COLLISION_MARGIN: float = 0.06  # should be >= RESUME_DIST
    MAX_PAUSE_STEPS: int = 500
    RESUME_DWELL: int = 15
    RAMP_STEPS: int = 50
    PRED_EVERY: int = 5  # 50Hz / 5 = 10Hz

    # Rendering
    render_mode: str = "rgb_array"   # env.render() frames
    highlight_visible_group: int = 2

    # Safety/robustness
    stop_on_error: bool = False
    resume_from_manifest: bool = True
    overwrite_existing: bool = False


def build_observation_config(cfg: DemoRecordConfig) -> ObservationConfig:
    return ObservationConfig(
        cameras=[
            CameraConfig(
                name=c.name,
                rgb=c.rgb,
                depth=c.depth,
                resolution=c.resolution,
            ) for c in cfg.cameras
        ],
        proprioception=cfg.proprioception,
        privileged_information=cfg.privileged_information,
    )


# ----------------------------
# Single-demo runner
# ----------------------------

def record_one_demo(cfg: DemoRecordConfig, demo_path: Path) -> Path:
    """
    Replays one demo with pause/resume collision handling and records a new
    demo file under cfg.save_root.
    Returns the saved demo file path.
    """
    demo_path = Path(demo_path)
    assert demo_path.exists(), f"Demo path invalid: {demo_path}"

    demo = Demo.from_safetensors(str(demo_path))

    # Save demos directly under save_root so DemoStore can discover them.
    out_dir = cfg.save_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # build envs
    observation_config = build_observation_config(cfg)

    action_mode = JointPositionActionMode(
        floating_base=cfg.floating_base,
        absolute=cfg.absolute,
        floating_dofs=list(cfg.floating_dofs),
    )

    env = HumanArmCupboardsOpenAll(
        action_mode=action_mode,
        observation_config=observation_config,
        render_mode=cfg.render_mode,
        arm_action_mode=cfg.arm_action_mode,
        control_frequency=cfg.control_frequency,
    )
    env.reset()

    # predictive env can skip obs_config for speed unless you need it
    env_pred = HumanArmCupboardsOpenAll(
        action_mode=action_mode,
        render_mode=cfg.render_mode,
        arm_action_mode=cfg.arm_action_mode,
        control_frequency=cfg.control_frequency,
    )
    env_pred.reset()

    recorder = DemoRecorder(str(out_dir))
    recorder.record(env, lightweight_demo=cfg.lightweight_demo)

    writer = None
    if cfg.write_demo_video:
        writer = imageio.get_writer(
            str(out_dir / f"{demo_path.stem}_replay.mp4"), fps=cfg.video_fps
        )
        frame_dt = 1.0 / cfg.video_fps
        sim_t = 0.0
        next_frame_t = 0.0

    # MuJoCo handles
    phys_pred = env_pred.mojo.physics
    m_pred = phys_pred.model.ptr

    # collision sets
    robot_ids_pred = collidable_ids_with_prefix(m_pred, "h1/")
    human_ids_pred = get_arm_geo_ids(m_pred)

    # apply margins in pred env
    set_margins_for_sets(env_pred.mojo.physics, robot_ids_pred, margin=cfg.COLLISION_MARGIN)
    set_margins_for_sets(env_pred.mojo.physics, human_ids_pred, margin=cfg.COLLISION_MARGIN)

    # highlighter (on rendered env but uses pred contact pair)
    hl = GeomHighlighter(
        env.mojo.physics,
        visible_group=cfg.highlight_visible_group,
        env=env,
        env_pred=env_pred,
    )

    # step limits
    n_timesteps = len(demo.timesteps)
    n_steps = n_timesteps if (cfg.n_steps is None or cfg.n_steps < 1) else min(n_timesteps, cfg.n_steps)

    # state buffer for fast copy
    buf = make_state_buffer(env.mojo.physics)

    # pause/ramp state
    t = 0
    demo_t = 0
    safe_count = 0
    ramp_k = cfg.RAMP_STEPS
    paused = False
    prev_paused = False
    pause_steps = 0
    resume_from_action = None
    last_safe_action = demo.timesteps[0].executed_action.copy()

    pbar = tqdm(total=n_steps, desc=f"Replaying: {demo_path.name}", dynamic_ncols=True)

    try:
        while t < n_steps:
            timestep = demo.timesteps[demo_t]
            proposed = timestep.executed_action.copy()

            c_hit = False
            cdist = np.inf
            cg1 = cg2 = -1

            # collision check
            if (t % cfg.PRED_EVERY == 0) or paused:
                copy_state(env, env_pred, buf)
                _ = env_pred.step(proposed)

                c_hit, cdist, cg1, cg2 = pair_min_contact_dist_between_sets(
                    phys_pred,
                    human_ids_pred,
                    robot_ids_pred,
                    dist_max=cfg.COLLISION_MARGIN,
                )

                pause_now = c_hit or (cdist < cfg.PAUSE_DIST)
                resume_ok = (not c_hit) and (cdist > cfg.RESUME_DIST)

                if (not paused) and pause_now:
                    paused = True
                    safe_count = 0
                    if cg1 != -1:
                        name1 = mujoco.mj_id2name(m_pred, mujoco.mjtObj.mjOBJ_GEOM, cg1) or ""
                        name2 = mujoco.mj_id2name(m_pred, mujoco.mjtObj.mjOBJ_GEOM, cg2) or ""
                    else:
                        name1 = name2 = ""
                    tqdm.write(f"[PAUSE] t={t} c_hit={c_hit} cdist={cdist:.4f} {name1} <-> {name2}")

                elif paused:
                    safe_count = (safe_count + 1) if resume_ok else 0
                    if safe_count >= cfg.RESUME_DWELL:
                        paused = False
                        safe_count = 0
                        tqdm.write(f"[RESUME] t={t} cdist={cdist:.4f}")

            # choose action
            if paused:
                hl.highlight_pred_contact_pair(cg1, cg2, rgba=(1, 0, 0, 1), highlight_body_visual=True)
                action = make_pause_hold_action(env, last_sent_action=last_safe_action)
                zero_floating_base_velocity(env)

                pause_steps += 1
                ramp_k = cfg.RAMP_STEPS
                resume_from_action = last_safe_action.copy()

                if pause_steps > cfg.MAX_PAUSE_STEPS:
                    paused = False
                    pause_steps = 0
                    tqdm.write(f"[FORCE RESUME] t={t} after {cfg.MAX_PAUSE_STEPS} pause steps")

            else:
                hl.clear()

                if prev_paused:
                    ramp_k = 0
                    if resume_from_action is None:
                        resume_from_action = last_safe_action.copy()

                if ramp_k < cfg.RAMP_STEPS:
                    alpha = (ramp_k + 1) / cfg.RAMP_STEPS
                    action = (1 - alpha) * resume_from_action + alpha * proposed
                    ramp_k += 1
                else:
                    action = proposed.copy()

                # only advance demo when not paused
                demo_t += 1
                last_safe_action = action.copy()
                pause_steps = 0

            # step main env + record
            output_timestep = env.step(action)
            recorder.add_timestep(output_timestep, action)

            # video
            if writer is not None:
                sim_t += env.get_dt()
                if sim_t >= next_frame_t:
                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("env.render() returned None; check render_mode.")
                    writer.append_data(frame)
                    next_frame_t += frame_dt

            # loop book-keeping
            t += 1
            prev_paused = paused
            pbar.update(1)
            pbar.set_postfix(paused=paused, cdist=float(cdist))

        saved_path = recorder.save_demo()
        if saved_path is None:
            raise RuntimeError("DemoRecorder failed to save demo.")
        return saved_path

    finally:
        pbar.close()
        recorder.stop()
        if writer is not None:
            writer.close()
        env.close()
        env_pred.close()


# ----------------------------
# Batch runner
# ----------------------------

def iter_demo_paths(demo_dir: Path, suffix: str = ".safetensors") -> list[Path]:
    demo_dir = Path(demo_dir)
    return sorted([p for p in demo_dir.rglob(f"*{suffix}") if p.is_file()])


def append_manifest_entry(manifest_path: Path, entry: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def load_completed_sources_from_manifest(manifest_path: Path) -> set[str]:
    completed = set()
    if not manifest_path.exists():
        return completed
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = str(entry.get("status", "")).upper()
            if status == "OK" and "source_demo" in entry:
                completed.add(entry["source_demo"])
    return completed


def run_batch(cfg: DemoRecordConfig, demo_paths: Sequence[Path]) -> dict[str, str]:
    """
    Runs batch recording with outer progress bar over demos.
    Returns dict: demo_path -> "OK:<out_dir>" or "ERR:<message>"
    """

    results: dict[str, str] = {}
    stats = Counter()
    manifest_path = Path(cfg.save_root) / "record_manifest.jsonl"

    demo_paths = [Path(p) for p in demo_paths]
    if cfg.resume_from_manifest and not cfg.overwrite_existing:
        completed_sources = load_completed_sources_from_manifest(manifest_path)
        demo_paths = [p for p in demo_paths if str(p) not in completed_sources]
    demo_paths = list(demo_paths)
    total = len(demo_paths)

    outer_bar = tqdm(
        demo_paths,
        total=total,
        desc="Recording demonstrations",
        dynamic_ncols=True,
        position=0,
    )

    for idx, p in enumerate(outer_bar, start=1):
        p = Path(p)

        try:
            saved_demo_path = record_one_demo(cfg, p)
            results[str(p)] = f"OK:{saved_demo_path}"
            stats["ok"] += 1
            append_manifest_entry(
                manifest_path,
                {
                    "status": "OK",
                    "source_demo": str(p),
                    "saved_demo": str(saved_demo_path),
                },
            )

        except Exception as e:
            results[str(p)] = f"ERR:{type(e).__name__}: {e}"
            stats["err"] += 1
            append_manifest_entry(
                manifest_path,
                {
                    "status": "ERR",
                    "error_type": type(e).__name__,
                    "source_demo": str(p),
                    "error": str(e),
                },
            )
            if cfg.stop_on_error:
                outer_bar.close()
                raise

        # Update outer progress display
        outer_bar.set_postfix(
            done=idx,
            total=total,
            ok=stats["ok"],
            err=stats["err"],
        )

    outer_bar.close()

    print("\nBatch summary:")
    print(f"  Total demos : {total}")
    print(f"  Successful  : {stats['ok']}")
    print(f"  Failed      : {stats['err']}")

    return results

# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    # Example 1: run on all demos in a folder
    demo_root = Path(f"{CACHE_PATH}",
                     "demonstrations",
                     "0.9.0",
                     "CupboardsOpenAll",
                     'JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute',
                     "lightweight")
    cfg = DemoRecordConfig(
        save_root=Path(f"{CACHE_PATH}",
                     "demonstrations",
                     "0.9.0",
                     "HumanArmCupboardsOpenAll",
                     'JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute',
                     "lightweight"),
        write_demo_video=False, # Dont write videos by default since they take up space and slow down recording; set True to enable
        n_steps=None,
        resume_from_manifest=True,   # default
        overwrite_existing=False,    # default
    )
    demo_paths = iter_demo_paths(demo_root, suffix=".safetensors")
    results = run_batch(cfg, demo_paths)
    print("\n".join([f"{k} -> {v}" for k, v in results.items()]))

    # Example 2: pull demos from DemoStore then select a subset
    # demo_store = DemoStore()
    # _ = demo_store.pull_demos()
    # demo_paths = [Path(".../demo1.safetensors"), Path(".../demo2.safetensors")]
    # results = run_batch(cfg, demo_paths)
