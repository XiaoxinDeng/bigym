"""An example of using BiGym with pixels."""
import numpy as np

from bigym.action_modes import TorqueActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.utils.observation_config import ObservationConfig, CameraConfig

try:
    from moviepy.editor import VideoClip
    import pygame  # noqa: F401
except ImportError:
    raise ImportError(
        "Please install moviepy and pygame for this example. "
        "i.e. `pip install moviepy pygame`"
    )


print("Running 1000 steps with pixels...")
env = ReachTarget(
    action_mode=TorqueActionMode(floating_base=True),
    observation_config=ObservationConfig(
        cameras=[
            CameraConfig(
                name="head",
                rgb=True,
                depth=False,
                resolution=(128, 128),
            )
        ],
    ),
    render_mode=None,
)

print("Observation Space:")
print(env.observation_space)
print("Action Space:")
print(env.action_space)

env.reset()
recorded_observations = []
for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    recorded_observations.append(obs["rgb_head"])
    if i % 1000 == 0:
        env.reset()
env.close()

frames = np.moveaxis(np.array(recorded_observations), 1, -1)
fps = 30
video_clip = VideoClip(
    make_frame=lambda t: frames[int(t * fps)], duration=int(len(frames) / fps)
)
video_clip.preview()
