import imageio
import numpy as np
from tqdm import tqdm, trange
from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards_with_human_arm import HumanArmCupboardsOpenAll
from bigym.envs.cupboards import CupboardsOpenAll
from demonstrations.utils import Metadata
from demonstrations.demo_store import DemoStore
from demonstrations.demo_player import DemoPlayer
from demonstrations.demo import Demo
from demonstrations.demo_converter import DemoConverter
from bigym.const import CACHE_PATH
from bigym.action_modes import PelvisDof
import os

control_frequency = 50

demo_env = CupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    render_mode="rgb_array",
    control_frequency=50,
)
metadata = Metadata.from_env(demo_env)

# Get demonstrations from DemoStore
demo_store = DemoStore()
try:
    demos = demo_store.get_demos(metadata, amount=1, frequency=control_frequency)
    demo = demos[0]
except Exception as e:
    print(e + "\n" + "==="*10)
    print(f"Run with a safetenser from CACHE_PATH: {CACHE_PATH}")
    filedir = f"{CACHE_PATH}/demonstrations/0.9.0/CupboardsOpenAll/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/"
    files = os.listdir(filedir)
    filename = os.path.join(filedir, files[0])
    assert os.path.exists(filename), f"{filename} invalid"
    demo = Demo.from_safetensors(filename)

n_steps = 3000
render = True
writer = imageio.get_writer("human_cupboard_demo.mp4", fps=30)

fps = 30
frame_dt = 1.0 / fps
sim_t = 0.0
next_frame_t = 0.0
floating_DOFS = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]

env = HumanArmCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=floating_DOFS),
    render_mode="rgb_array",
    arm_action_mode="position",
    control_frequency=50,
)
env.reset()



# # Replay first demonstration
# player = DemoPlayer()
# player.replay_in_env(demo, env, demo_frequency=control_frequency)
request = []
actual = []

# demo = DemoConverter.absolute_to_delta(demo)

# Replay the demo
for timestep in tqdm(demo.timesteps, desc="Processing timesteps"):    
    # Using joint positions as action does not reproduce the same trajectory
    # since the simulation is controlled using PID controllers.
    action = timestep.executed_action
    obs, reward, termination, truncation, info = env.step(action)
    request.append(action)
    actual.append(env.robot.qpos_actuated)
    for key, val in timestep.observation.items():
        assert np.allclose(val, obs[key], atol=1e-6), f"Key: {key}"

    sim_t += env.get_dt()  # env_dt = opt.timestep * substeps
    if sim_t >= next_frame_t:
        frame = env.render()
        if frame is None: raise RuntimeError("env.render() returned None; use direct MuJoCo rendering (Section C).")
        writer.append_data(frame)
        next_frame_t += frame_dt

writer.close()
env.close()

