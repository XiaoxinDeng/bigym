import imageio
import numpy as np
from tqdm import tqdm, trange
from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards_with_human import HumanCupboardsOpenAll

env = HumanCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    render_mode="rgb_array",
)
env.reset()

writer = imageio.get_writer("human_cupboard.mp4", fps=30)

n_steps = 300
for t in trange(n_steps):
    action = np.zeros_like(env.action_space.sample())
    obs, reward, termination, truncation, info = env.step(action)

    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None; use direct MuJoCo rendering (Section C).")
    writer.append_data(frame)

writer.close()
env.close()
