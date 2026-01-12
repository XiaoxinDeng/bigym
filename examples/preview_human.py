import imageio
import numpy as np
from tqdm import trange
from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards_with_human import HumanCupboardsOpenAll
from bigym.utils.observation_config import ObservationConfig, CameraConfig

PREVIEW = False  # set True only when needed
SAVE_FRAMES = True  # set True only when needed
FPS = 30
N_STEPS = 600
CAM_NAME = "external2"
OUTPUT_VIDEO_FILE = "human_cupboard.test.mp4"
W, H = 256, 256
if PREVIEW:
    try:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("BiGym Live Preview (press Q to quit)")
        clock = pygame.time.Clock()
    except ImportError:
        raise ImportError("Please install pygame: `pip install pygame`")

env = HumanCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    observation_config=ObservationConfig(
       cameras=[
            CameraConfig(
                name=CAM_NAME,
                rgb=True,
                depth=False,
                resolution=(W, H),
            ),
       ],
    ),
    render_mode="rgb_array",
)
env.reset()

if SAVE_FRAMES:
    writer = imageio.get_writer(OUTPUT_VIDEO_FILE, fps=FPS)

for t in trange(N_STEPS):
    action = np.zeros_like(env.action_space.sample())
    obs, reward, termination, truncation, info = env.step(action)

    frame = obs["rgb_" + CAM_NAME]  # (3, H, W), uint8, RGB
    frame = np.moveaxis(frame, 0, -1)   # (H, W, 3)

    if frame is None:
        raise RuntimeError("env.render() returned None; use direct MuJoCo rendering (Section C).")
    
    if SAVE_FRAMES:
        writer.append_data(frame)
    
    if PREVIEW:
        # # Preview: pygame expects (W, H, 3)
        # surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        # screen.blit(surf, (0, 0))
        # pygame.display.flip()
        # # pace preview (does not change simulation speed if sim is slower)
        # clock.tick(FPS)
        # Process events so the window actually refreshes
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        # Ensure uint8 RGB HWC
        frame_hwc = frame
        if frame_hwc.dtype != np.uint8:
            frame_hwc = np.clip(frame_hwc, 0, 255).astype(np.uint8)

        # Pygame surfarray expects (W, H, 3), so transpose HWC -> WHC
        frame_whc = np.ascontiguousarray(frame_hwc.transpose(1, 0, 2))

        # If you *still* see black, try uncommenting this to test BGR/RGB mismatch:
        # frame_whc = frame_whc[..., ::-1]

        pygame.surfarray.blit_array(screen, frame_whc)
        pygame.display.flip()
        clock.tick(FPS)
    
    if termination or truncation:
        obs, _ = env.reset()

if SAVE_FRAMES:
    writer.close()
    print(f"Saved video to {OUTPUT_VIDEO_FILE}")
if PREVIEW:
    pygame.quit()
env.close()
