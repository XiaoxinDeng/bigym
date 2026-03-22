import csv
import json
from pathlib import Path

from posixpath import sep
import imageio
import numpy as np
from tqdm import tqdm
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.envs.cupboards import DrawerTopOpen
from demonstrations.demo_store import DemoStore
from demonstrations.demo import Demo
from bigym.const import CACHE_PATH
import os

from demo_utils import *
from demo_utils import GeomHighlighter
from demonstrations.demo_recorder import DemoRecorder
from bigym.utils.observation_config import ObservationConfig, CameraConfig


CLASS_NAME = "DrawerTopOpen"

# Get demonstrations from DemoStore In case users do not have demos installed
demo_store = DemoStore()
demos = demo_store.pull_demos()

# Load demo from CACHE
filedir = f"{CACHE_PATH}/demonstrations/0.9.0/"
filedir2 = "JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/"
files = os.listdir(os.path.join(filedir, CLASS_NAME, filedir2))

# Set variables
n_demo_steps = None
render = False
write_demo_video = False
save_demo_to_disk = False
control_frequency = 50
fps = 30
frame_dt = 1.0 / fps

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

env = DrawerTopOpen(
    action_mode=JointPositionActionMode(
        floating_base=True,
        absolute=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    ),
    observation_config=observation_config,
    render_mode="rgb_array",
    control_frequency=control_frequency,
)

# -------------------------
# Manifest output paths
# -------------------------
save_dir = os.path.join(filedir, "HumanArm", f"HumanArm{CLASS_NAME}", filedir2)
os.makedirs(save_dir, exist_ok=True)

manifest_json = os.path.join(filedir, CLASS_NAME, filedir2, f"manifest.json")

manifest = []

for file in files:
    filename = os.path.join(filedir, CLASS_NAME, filedir2, file)
    assert os.path.exists(filename), f"Demo path invalid: {filename}"

    demo = Demo.from_safetensors(filename)
    uuid = demo.uuid

    if write_demo_video:
        writer = imageio.get_writer(f"cupboard_{CLASS_NAME}_demo_{uuid}.mp4", fps=30)

    n_timesteps = len(demo.timesteps)
    if n_demo_steps is None or n_demo_steps < 1:
        cur_demo_steps = n_timesteps
    else:
        cur_demo_steps = min(n_timesteps, n_demo_steps)

    obs = env.reset()

    recorder = DemoRecorder(save_dir)
    if save_demo_to_disk:
        recorder.record(env, lightweight_demo=True)

    sim_t = 0.0
    next_frame_t = 0.0
    demo_t = 0

    success = False
    final_drawer_state = None
    terminated = False
    truncated = False
    exception_msg = ""

    pbar = tqdm(
        total=cur_demo_steps,
        initial=demo_t,
        desc=f"Replaying demo {uuid}",
        dynamic_ncols=True,
    )

    try:
        while demo_t < cur_demo_steps:
            timestep = demo.timesteps[demo_t]
            proposed = timestep.executed_action.copy()

            obs, reward, terminated, truncated, info = env.step(proposed)
            success = bool(env.success)

            if save_demo_to_disk:
                recorder.add_timestep((obs, reward, terminated, truncated, info), proposed)

            demo_t += 1
            pbar.update()

            sim_t += env.get_dt()
            if sim_t >= next_frame_t and write_demo_video:
                frame = env.render()
                if frame is None:
                    raise RuntimeError("env.render() returned None; use direct MuJoCo rendering.")
                writer.append_data(frame)
                next_frame_t += frame_dt

        final_drawer_state = env.cabinet_drawers.get_state()[-1]

    except Exception as e:
        exception_msg = repr(e)
        success = False

    finally:
        pbar.close()
        if save_demo_to_disk:
            recorder.save_demo()
            recorder.stop()
        if write_demo_video:
            writer.close()

    record = {
        "task": CLASS_NAME,
        "uuid": uuid,
        "filename": file,
        "full_path": filename,
        "num_timesteps": n_timesteps,
        "num_replayed_steps": demo_t,
        "success": int(success),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "final_drawer_state": (
            float(final_drawer_state) if final_drawer_state is not None else None
        ),
        "exception": exception_msg,
    }
    manifest.append(record)

    print(f"Inspecting file: {uuid}")
    print(f"Success: {success} | final drawer state: {final_drawer_state}")

env.close()

# -------------------------
# Save manifest as CSV
# -------------------------
fieldnames = [
    "task",
    "uuid",
    "filename",
    "full_path",
    "num_timesteps",
    "num_replayed_steps",
    "success",
    "terminated",
    "truncated",
    "final_drawer_state",
    "exception",
]

# -------------------------
# Save manifest as JSON
# -------------------------
with open(manifest_json, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"Saved JSON manifest to: {manifest_json}")