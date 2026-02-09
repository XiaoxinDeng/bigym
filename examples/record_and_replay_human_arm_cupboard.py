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


def _snapshot(physics):
    # Use raw MuJoCo pointers from dm_control wrapper
    model = physics.model.ptr
    data = physics.data.ptr
    n = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state = np.zeros(n, dtype=np.float64)
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return state

def _restore(physics, state):
    model = physics.model.ptr
    data = physics.data.ptr
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, data)

def _robot_colliders(env):
    geoms = []
    for geom_mjcf in env.robot._body.mjcf.find_all("geom"):
        g = Geom(env.mojo, geom_mjcf)
        if g.is_collidable():
            geoms.append(g)
    return geoms

def _hold_action(env):
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    base_n = env.robot.floating_base.dof_amount if env.action_mode.floating_base else 0
    limb_n = len(env.robot.limb_actuators)
    grip_n = len(env.robot.grippers)

    if isinstance(env.action_mode, JointPositionActionMode) and env.action_mode.absolute:
        qpos = env.robot.qpos_actuated
        # Base is always delta → keep it zero
        action[base_n:base_n + limb_n] = qpos[base_n:base_n + limb_n]
        # Grippers are position‑controlled
        if grip_n:
            action[-grip_n:] = qpos[-grip_n:]
    else:
        # Delta/torque: keep zero motion, but hold grippers at current position
        if grip_n:
            action[-grip_n:] = env.robot.qpos_grippers

    return action

def will_collide_within(env:_HumanArmCupboardsInteractionEnv, 
                        robot_colliders, horizon_s, step_dt=None):
    physics = env.mojo.physics
    model = physics.model.ptr
    data = physics.data.ptr
    human = env.humanarms[0]  # _HUMAN_COUNT is 1 in cupboards_with_human_arm.py
    step_dt = float(step_dt or env.get_dt())

    steps = int(np.ceil(horizon_s / step_dt))
    state = _snapshot(physics)
    t0 = human._CURRENT_TIME

    try:
        for i in range(steps):
            t = t0 + (i + 1) * step_dt
            q_des = human._traj(t)  # scripted harmonic pose

            # Directly set human joints in the lookahead state (fast, deterministic)
            physics.data.qpos[human._qpos_adr] = q_des
            physics.data.qvel[human._qvel_adr] = 0.0
            mujoco.mj_forward(model, data)

            if has_collided_collections(physics, human.colliders, robot_colliders):
                return True, (i + 1) * step_dt
        return False, None
    finally:
        _restore(physics, state)



env = HumanArmCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, 
                                        absolute=True, 
                                        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]),
    render_mode="rgb_array",
    arm_action_mode="scripted",
    control_frequency=50,
)
env.reset()
robot_colliders = _robot_colliders(env)
horizon_s = 0.6  # collision lookahead threshold in seconds

# request = []
# actual = []

# Replay the demo
for t in trange(n_steps, desc="Processing timesteps"):    
    # Using joint positions as action does not reproduce the same trajectory
    # since the simulation is controlled using PID controllers.
    timestep = demo.timesteps[t]
    action = timestep.executed_action
    will_hit, ttc = will_collide_within(env, robot_colliders, horizon_s)
    if will_hit:
        action = _hold_action(env)  # pause robot if collision predicted

    obs, reward, termination, truncation, info = env.step(action)
    # request.append(action)
    # actual.append(env.robot.qpos_actuated)
    for key, val in timestep.observation.items():
        assert np.allclose(val, obs[key], atol=1e-6), f"Key: {key}"

    # Write frame in simulation time
    sim_t += env.get_dt()  # env_dt = opt.timestep * substeps
    if sim_t >= next_frame_t:
        frame = env.render()
        if frame is None: raise RuntimeError("env.render() returned None; use direct MuJoCo rendering (Section C).")
        writer.append_data(frame)
        next_frame_t += frame_dt

writer.close()
env.close()

