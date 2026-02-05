import imageio
import numpy as np
from tqdm import tqdm, trange
from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards_with_human_arm import HumanArmCupboardsOpenAll

base_min, base_max = -1.570796,  1.570796
yaw_min, yaw_max = -1.570796,  1.570796
pit_min, pit_max = -1.570796,  1.570796
elb_min, elb_max =  0.0,       2.530727

def sin_to_range(s, lo, hi, margin=0.05):
    lo2 = lo + margin
    hi2 = hi - margin
    mid = 0.5 * (lo2 + hi2)
    amp = 0.5 * (hi2 - lo2)
    return mid + amp * s

w = 2*np.pi*0.4


env = HumanArmCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    render_mode="rgb_array",
    arm_action_mode="position",
)
env.reset()

writer = imageio.get_writer("human_cupboard.mp4", fps=30)
n_steps = 3000
fps = 30
frame_dt = 1.0 / fps
sim_t = 0.0
next_frame_t = 0.0

for t in trange(n_steps):
    action = np.zeros_like(env.action_space.sample())
    arm_action = np.array([
        0,
        # sin_to_range(np.sin(w*sim_t + 3.0), base_min, base_max),
        sin_to_range(np.sin(w*sim_t + 0.0), yaw_min, yaw_max),
        sin_to_range(np.sin(w*sim_t + 1.0), pit_min, pit_max),
        sin_to_range(np.sin(w*sim_t + 2.0), elb_min, elb_max),   # IMPORTANT: elbow is [0,2.53]
    ], dtype=np.float64)


    obs, reward, termination, truncation, info = env.step(action, arm_action=arm_action)
    sim_t += env.get_dt()  # env_dt = opt.timestep * substeps
    if sim_t >= next_frame_t:
        frame = env.render()
        if frame is None: raise RuntimeError("env.render() returned None; use direct MuJoCo rendering (Section C).")
        writer.append_data(frame)
        next_frame_t += frame_dt

writer.close()
env.close()
