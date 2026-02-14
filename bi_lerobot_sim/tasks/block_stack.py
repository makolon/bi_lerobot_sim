"""Block stacking task."""

import copy
import os

from bi_lerobot_sim.tasks import lerobot_task
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
import numpy as np

_BLOCK_RESET_HEIGHT = 0.02


# Red block position (to be placed on top)
red_block_uniform_position = distributions.Uniform(
    low=[-0.04, 0.08, _BLOCK_RESET_HEIGHT],
    high=[0.04, 0.16, _BLOCK_RESET_HEIGHT],
    single_sample=True,
)

# Blue block position (base block - bottom)
blue_block_uniform_position = distributions.Uniform(
    low=[0.41, 0.08, _BLOCK_RESET_HEIGHT],
    high=[0.49, 0.16, _BLOCK_RESET_HEIGHT],
    single_sample=True,
)

# Random rotation around Z-axis
block_z_rotation = rotations.UniformQuaternion()


class BlockStack(lerobot_task.LeRobotTask):
    """Stack blocks in 2 levels: blue (bottom), red (top)."""

    def __init__(
        self,
        red_block_path: str | None = None,
        blue_block_path: str | None = None,
        **kwargs,
    ):
        """Initializes a new `BlockStack` task.

        Args:
            red_block_path: Path to asset of the red block (top).
            blue_block_path: Path to asset of the blue block (bottom/base).
            **kwargs: Additional args to pass to the base class.
        """
        super().__init__(
            **kwargs,
        )

        assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    
        # Default block paths if not provided
        if red_block_path is None:
            red_block_path = os.path.join(assets_dir, 'blocks', 'red_block.xml')
        if blue_block_path is None:
            blue_block_path = os.path.join(assets_dir, 'blocks', 'blue_block.xml')

        # Try to load blocks; skip if they don't load properly
        self._red_block_prop = None
        self._blue_block_prop = None
        self._block_placers = []
        
        # Load red block (top)
        self._red_block_prop = composer.ModelWrapperEntity(
            mjcf.from_path(red_block_path)
        )
        self._scene.add_free_entity(self._red_block_prop)

        # Load blue block (bottom)
        self._blue_block_prop = composer.ModelWrapperEntity(
            mjcf.from_path(blue_block_path)
        )
        self._scene.add_free_entity(self._blue_block_prop)

        # Remove freejoints to use PropPlacer
        for prop in [self._red_block_prop, self._blue_block_prop]:
            freejoint = traversal_utils.get_freejoint(
                prop.mjcf_model.find_all('body')[0]
            )
            if freejoint:
                freejoint.remove()

        # Adjust positions for table height offset
        red_block_position = copy.deepcopy(red_block_uniform_position)
        blue_block_position = copy.deepcopy(blue_block_uniform_position)

        # Create prop placers for blocks (bottom to top order)
        self._block_placers = [
            initializers.PropPlacer(
                props=[self._blue_block_prop],
                position=blue_block_position,
                quaternion=block_z_rotation,
                ignore_collisions=True,
                settle_physics=False,
                max_attempts_per_prop=100,
            ),
            initializers.PropPlacer(
                props=[self._red_block_prop],
                position=red_block_position,
                quaternion=block_z_rotation,
                ignore_collisions=True,
                max_attempts_per_prop=100,
                settle_physics=False,
            ),
            initializers.PropPlacer(
                props=[self._red_block_prop, self._blue_block_prop],
                position=deterministic.Identity(),
                quaternion=deterministic.Identity(),
                ignore_collisions=True,  # Collisions already resolved.
                settle_physics=True,
            ),
        ]

        # Update qpos for the three blocks (each has 7 DOF: 3 pos + 4 quat)
        extra_qpos = np.zeros((21,))

        scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
        if scene_key is not None:
            scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for prop_placer in self._block_placers:
            prop_placer(physics, random_state)

    def get_reward(self, physics):
        """Returns 1.0 when blocks are stacked; otherwise 0.0."""
        red_body = self._red_block_prop.mjcf_model.find_all('body')[0]
        blue_body = self._blue_block_prop.mjcf_model.find_all('body')[0]
        red_pos = physics.bind(red_body).xpos
        blue_pos = physics.bind(blue_body).xpos

        red_geom = self._red_block_prop.mjcf_model.find_all('geom')[0]
        blue_geom = self._blue_block_prop.mjcf_model.find_all('geom')[0]

        red_half = np.array(red_geom.size, dtype=np.float32)
        blue_half = np.array(blue_geom.size, dtype=np.float32)

        target_height = float(red_half[2] + blue_half[2])
        xy_tolerance = float(min(red_half[0], red_half[1], blue_half[0], blue_half[1]))
        z_tolerance = float(min(red_half[2], blue_half[2]) * 0.5)

        xy_error = float(np.linalg.norm(red_pos[:2] - blue_pos[:2]))
        z_error = float(abs((red_pos[2] - blue_pos[2]) - target_height))

        stacked = (xy_error <= xy_tolerance) and (z_error <= z_tolerance)
        return 1.0 if stacked else 0.0
