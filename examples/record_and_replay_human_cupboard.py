"""An example of demo collection using BiGym with pixels."""
import matplotlib.pyplot as plt
import numpy as np
import tempfile

from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards_with_human import HumanCupboardsOpenAll
from demonstrations.demo_recorder import DemoRecorder
from demonstrations.demo_converter import DemoConverter
from demonstrations.demo import Demo

render = False
cam = "head"
cam_key = f"rgb_{cam}"

env = HumanCupboardsOpenAll(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    render_mode="human" if render else None,
)


def update_plots(axs, requested, expected, actual, xlim=None, ylim=None):
    """Update plots."""
    for i, (r, e, a) in enumerate(zip(requested, expected, actual)):
        axs[i].clear()  # Clear previous plot
        axs[i].plot(r, label="Request")
        axs[i].plot(e, label="Expected")
        axs[i].plot(a, label="Actual")
        axs[i].set_title(f"Variable {i}")
        axs[i].legend()
        if xlim is not None:
            axs[i].set_xlim(xlim)
        if ylim is not None:
            axs[i].set_ylim(ylim)

    plt.tight_layout()


def init_subplots(n):
    """Initialize subplots."""
    fig = plt.figure(figsize=(8, 2 * n))
    axs = [fig.add_subplot(n, 1, i + 1) for i in range(n)]
    return fig, axs


with tempfile.TemporaryDirectory() as temp_dir:
    demo_recorder = DemoRecorder(temp_dir)

    amplitude = 0.2
    frequency = 0.02
    episode_length = 500

    # Record the demo
    env.reset()
    demo_recorder.record(env)

    expected = []

    for i in range(episode_length):
        # Only move elbow joints
        action = np.zeros_like(env.action_space.sample())
        action[6] = amplitude * np.sin(frequency * i)
        action[10] = amplitude * -np.sin(frequency * i)
        action[:3] = np.zeros(3)
        timestep = env.step(action)
        demo_recorder.add_timestep(timestep, action)
        expected.append(env.robot.qpos_actuated)
        if render:
            env.render()
    demo_recorder.stop()
    env.close()

    # Save the time steps to a safetensors
    filepath = demo_recorder.save_demo()

    # Load the time steps from a safetensors
    demo = Demo.from_safetensors(filepath)

    env = HumanCupboardsOpenAll(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
        render_mode="human" if render else None,
    )
    env.reset(seed=demo.seed)

    request = []
    actual = []

    demo = DemoConverter.absolute_to_delta(demo)

    # Replay the demo
    for timestep in demo.timesteps:
        # Using joint positions as action does not reproduce the same trajectory
        # since the simulation is controlled using PID controllers.
        action = timestep.executed_action
        obs, reward, termination, truncation, info = env.step(action)
        request.append(action)
        actual.append(env.robot.qpos_actuated)
        for key, val in timestep.observation.items():
            assert np.allclose(val, obs[key], atol=1e-6), f"Key: {key}"
        if render:
            env.render()

    # Update the plots
    request = np.array(request).T
    expected = np.array(expected).T
    actual = np.array(actual).T
    fig, axs = init_subplots(len(env.action_space.sample()))
    amplitude = abs(actual).max()
    update_plots(
        axs,
        request,
        expected,
        actual,
        ylim=(-amplitude * 1.1, amplitude * 1.1),
    )
    plt.show()
