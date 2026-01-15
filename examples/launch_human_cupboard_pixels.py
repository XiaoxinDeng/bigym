"""An example of using BiGym with pixels."""
import numpy as np

from bigym.action_modes import TorqueActionMode
from bigym.envs.cupboards_with_human import HumanCupboardsCloseAll
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from tqdm import trange
try:
    from moviepy.editor import VideoClip
    import pygame  # noqa: F401
except ImportError:
    raise ImportError(
        "Please install moviepy and pygame for this example. "
        "i.e. `pip install moviepy pygame`"
    )

n_steps = 300
print(f"Running {n_steps} steps with pixels...")
env = HumanCupboardsCloseAll(
    action_mode=TorqueActionMode(floating_base=True),
    observation_config=ObservationConfig(
        cameras=[
            # CameraConfig(
            #     name="head",
            #     rgb=True,
            #     depth=False,
            #     resolution=(128, 128),
            # ),
            CameraConfig(
                name="external", 
                rgb=True, 
                depth=False, 
                resolution=(256, 256)
            ),
        ],
    ),
    render_mode="rgb_array",
)

print("Observation Space:")
print(env.observation_space)
print("Action Space:")
print(env.action_space)

env.reset()
recorded_observations = []
for i in trange(n_steps):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # recorded_observations.append(obs["rgb_head"])
    recorded_observations.append(obs["rgb_external"])
    if i % n_steps == 0:
        env.reset()
env.close()

frames = np.moveaxis(np.array(recorded_observations), 1, -1)
fps = 30
video_clip = VideoClip(
    make_frame=lambda t: frames[int(t * fps)], duration=int(len(frames) / fps)
)
print("Previewing video...")
video_clip.preview()
print("Saving video to human_cupboard.mp4...")
video_clip.write_videofile("human_cupboard.mp4", fps)
