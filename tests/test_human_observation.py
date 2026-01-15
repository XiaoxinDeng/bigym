"""Tests for human observation functionality."""
import numpy as np
import pytest

from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards_with_human import HumanCupboardsOpenAll
from bigym.utils.observation_config import ObservationConfig


class TestHumanObservation:
    """Test suite for human observation with movement."""

    @pytest.fixture
    def env_with_human_obs(self):
        """Create environment with privileged human observations enabled."""
        obs_config = ObservationConfig(privileged_information=True)
        env = HumanCupboardsOpenAll(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            observation_config=obs_config,
            render_mode=None,
        )
        yield env
        env.close()

    @pytest.fixture
    def env_without_human_obs(self):
        """Create environment without privileged observations."""
        obs_config = ObservationConfig(privileged_information=False)
        env = HumanCupboardsOpenAll(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            observation_config=obs_config,
            render_mode=None,
        )
        yield env
        env.close()

    def test_human_joints_in_observation(self, env_with_human_obs):
        """Test that human_joints key exists in observation when privileged_information=True."""
        obs, info = env_with_human_obs.reset(seed=42)
        assert "human_joints" in obs, "human_joints should be in observation"

    def test_human_joints_not_in_observation_by_default(self, env_without_human_obs):
        """Test that human_joints is not in observation when privileged_information=False."""
        obs, info = env_without_human_obs.reset(seed=42)
        assert "human_joints" not in obs, "human_joints should not be in default observation"

    def test_human_joints_shape(self, env_with_human_obs):
        """Test that human_joints has correct shape (num_joints * 3)."""
        obs, info = env_with_human_obs.reset(seed=42)
        human = env_with_human_obs.humans[0]
        expected_shape = (human.num_joints * 3,)
        assert obs["human_joints"].shape == expected_shape, (
            f"Expected shape {expected_shape}, got {obs['human_joints'].shape}"
        )

    def test_human_joints_dtype(self, env_with_human_obs):
        """Test that human_joints is a float type."""
        obs, info = env_with_human_obs.reset(seed=42)
        assert np.issubdtype(obs["human_joints"].dtype, np.floating), (
            f"Expected floating point dtype, got {obs['human_joints'].dtype}"
        )

    def test_human_movement_over_time(self, env_with_human_obs):
        """Test that human joints change over multiple steps (human is moving)."""
        obs, info = env_with_human_obs.reset(seed=42)
        initial_joints = obs["human_joints"].copy()

        # Run enough steps to ensure at least one frame change (~16 steps per frame at 30fps)
        for _ in range(100):
            action = np.zeros_like(env_with_human_obs.action_space.sample())
            obs, _, _, _, _ = env_with_human_obs.step(action)

        final_joints = obs["human_joints"]

        # Joints should have changed
        assert not np.allclose(initial_joints, final_joints, atol=1e-6), (
            "Human joints should change over time (human should be moving)"
        )

    def test_human_movement_frame_rate(self, env_with_human_obs):
        """Test that human moves at expected frame rate (~16 steps per frame)."""
        obs, info = env_with_human_obs.reset(seed=42)
        prev_joints = obs["human_joints"].copy()

        frame_changes = 0
        steps_between_changes = []
        last_change_step = 0

        for step in range(200):
            action = np.zeros_like(env_with_human_obs.action_space.sample())
            obs, _, _, _, _ = env_with_human_obs.step(action)

            if not np.allclose(obs["human_joints"], prev_joints, atol=1e-6):
                frame_changes += 1
                if frame_changes > 1:
                    steps_between_changes.append(step - last_change_step)
                last_change_step = step
                prev_joints = obs["human_joints"].copy()

        # Should have multiple frame changes
        assert frame_changes >= 10, f"Expected at least 10 frame changes, got {frame_changes}"

        # Steps between changes should be roughly consistent (~16-17 steps)
        if steps_between_changes:
            avg_steps = np.mean(steps_between_changes)
            assert 14 <= avg_steps <= 20, (
                f"Expected ~16-17 steps between frames, got avg {avg_steps:.1f}"
            )

    def test_seed_reproducibility(self, env_with_human_obs):
        """Test that same seed produces same initial human joint positions."""
        obs1, _ = env_with_human_obs.reset(seed=42)
        joints1 = obs1["human_joints"].copy()

        obs2, _ = env_with_human_obs.reset(seed=42)
        joints2 = obs2["human_joints"].copy()

        assert np.allclose(joints1, joints2, atol=1e-6), (
            "Same seed should produce same initial joint positions"
        )

    def test_different_seeds_different_positions(self, env_with_human_obs):
        """Test that different seeds produce different initial positions."""
        obs1, _ = env_with_human_obs.reset(seed=42)
        joints1 = obs1["human_joints"].copy()

        obs2, _ = env_with_human_obs.reset(seed=123)
        joints2 = obs2["human_joints"].copy()

        assert not np.allclose(joints1, joints2, atol=1e-6), (
            "Different seeds should produce different initial positions"
        )

    def test_joint_positions_reasonable_range(self, env_with_human_obs):
        """Test that joint positions are within reasonable physical bounds."""
        obs, info = env_with_human_obs.reset(seed=42)
        joints = obs["human_joints"]

        # Reshape to (num_joints, 3) for easier analysis
        num_joints = len(joints) // 3
        joints_reshaped = joints.reshape(num_joints, 3)

        # Check reasonable bounds (human should be near origin, within ~2m)
        assert np.all(np.abs(joints_reshaped[:, 0]) < 3.0), "X positions out of range"
        assert np.all(np.abs(joints_reshaped[:, 1]) < 3.0), "Y positions out of range"
        assert np.all(joints_reshaped[:, 2] > 0.0), "Z positions should be positive (above ground)"
        assert np.all(joints_reshaped[:, 2] < 3.0), "Z positions out of range"

    def test_human_motion_continuity(self, env_with_human_obs):
        """Test that human motion is continuous (no large jumps between frames)."""
        obs, info = env_with_human_obs.reset(seed=42)
        prev_joints = obs["human_joints"].copy()

        max_jump = 0.0
        for _ in range(100):
            action = np.zeros_like(env_with_human_obs.action_space.sample())
            obs, _, _, _, _ = env_with_human_obs.step(action)

            # Calculate max position change
            diff = np.abs(obs["human_joints"] - prev_joints)
            max_jump = max(max_jump, diff.max())
            prev_joints = obs["human_joints"].copy()

        # Motion should be smooth - no jumps larger than 0.5m per step
        assert max_jump < 0.5, f"Motion discontinuity detected: max jump = {max_jump:.3f}m"


class TestHumanProperties:
    """Test human model properties."""

    @pytest.fixture
    def env(self):
        """Create environment."""
        obs_config = ObservationConfig(privileged_information=True)
        env = HumanCupboardsOpenAll(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            observation_config=obs_config,
            render_mode=None,
        )
        yield env
        env.close()

    def test_human_exists(self, env):
        """Test that human object exists in environment."""
        env.reset(seed=42)
        assert hasattr(env, "humans"), "Environment should have humans attribute"
        assert len(env.humans) > 0, "Should have at least one human"

    def test_human_num_joints(self, env):
        """Test human has expected number of joints."""
        env.reset(seed=42)
        human = env.humans[0]
        assert human.num_joints > 0, "Human should have joints"
        # Typical mocap skeleton has 20-150 joints
        assert 10 < human.num_joints < 200, f"Unexpected joint count: {human.num_joints}"

    def test_human_fps(self, env):
        """Test human motion FPS is reasonable."""
        env.reset(seed=42)
        human = env.humans[0]
        assert human.fps > 0, "FPS should be positive"
        # Typical mocap is 30-120 FPS
        assert 10 <= human.fps <= 240, f"Unexpected FPS: {human.fps}"

    def test_human_num_frames(self, env):
        """Test human has motion frames."""
        env.reset(seed=42)
        human = env.humans[0]
        assert human._num_frames > 0, "Should have motion frames"

    def test_human_get_joint_positions(self, env):
        """Test get_joint_positions returns correct shape."""
        env.reset(seed=42)
        human = env.humans[0]
        positions = human.get_joint_positions()

        expected_shape = (human.num_joints * 3,)
        assert positions.shape == expected_shape, (
            f"Expected {expected_shape}, got {positions.shape}"
        )

    def test_human_colliders_exist(self, env):
        """Test that human has collision geometry."""
        env.reset(seed=42)
        human = env.humans[0]
        assert hasattr(human, "colliders"), "Human should have colliders"
        assert len(human.colliders) > 0, "Human should have at least one collider"


class TestHumanObservationVisual:
    """Visual/rendering tests for human observation (can be skipped in CI)."""

    @pytest.fixture
    def env_with_render(self):
        """Create environment with rendering."""
        obs_config = ObservationConfig(privileged_information=True)
        env = HumanCupboardsOpenAll(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            observation_config=obs_config,
            render_mode=None,  # Set to "human" for visual testing
        )
        yield env
        env.close()

    def test_extended_movement_tracking(self, env_with_render):
        """Test tracking human movement over extended period."""
        obs, info = env_with_render.reset(seed=42)

        joint_trajectory = [obs["human_joints"].copy()]

        # Run for 500 steps (~1 second)
        for _ in range(500):
            action = np.zeros_like(env_with_render.action_space.sample())
            obs, _, _, _, _ = env_with_render.step(action)
            joint_trajectory.append(obs["human_joints"].copy())

        trajectory = np.array(joint_trajectory)

        # Calculate total displacement
        total_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])

        # Human should have moved noticeably
        assert total_displacement > 0.1, (
            f"Human should move significantly over 1 second, got {total_displacement:.4f}m"
        )

        # Calculate per-joint displacements
        num_joints = trajectory.shape[1] // 3
        joint_displacements = []
        for j in range(num_joints):
            start = trajectory[0, j*3:(j+1)*3]
            end = trajectory[-1, j*3:(j+1)*3]
            joint_displacements.append(np.linalg.norm(end - start))

        jd = np.array(joint_displacements)

        # Different joints should move different amounts (articulated motion)
        assert jd.std() > 0.01, "Motion should be articulated (different joints move differently)"
