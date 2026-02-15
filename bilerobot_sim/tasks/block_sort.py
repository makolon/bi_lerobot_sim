"""Block sorting task."""

import os

from bilerobot_sim.tasks import lerobot_task
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
import numpy as np

_BLOCK_RESET_HEIGHT = 0.02
_TARGET_RADIUS = 0.05

# Blue block position (start)
blue_block_uniform_position = distributions.Uniform(
    low=[-0.04, 0.02, _BLOCK_RESET_HEIGHT],
    high=[0.04, 0.10, _BLOCK_RESET_HEIGHT],
    single_sample=True,
)

# Red block position (start)
red_block_uniform_position = distributions.Uniform(
    low=[0.41, 0.02, _BLOCK_RESET_HEIGHT],
    high=[0.49, 0.10, _BLOCK_RESET_HEIGHT],
    single_sample=True,
)

# Target positions
BLUE_TARGET_POS = np.array([0.0, 0.18, 0.0])
RED_TARGET_POS = np.array([0.45, 0.18, 0.0])

# Random rotation around Z-axis
block_z_rotation = rotations.UniformQuaternion()


class BlockSort(lerobot_task.LeRobotTask):
    """Sort blue and red blocks into target zones."""

    def __init__(
        self,
        blue_block_path: str | None = None,
        red_block_path: str | None = None,
        **kwargs,
    ):
        """Initializes a new `BlockSort` task.

        Args:
            blue_block_path: Path to asset of the blue block.
            red_block_path: Path to asset of the red block.
            **kwargs: Additional args to pass to the base class.
        """
        super().__init__(
            **kwargs,
        )

        assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    
        # Default block paths if not provided
        if blue_block_path is None:
            blue_block_path = os.path.join(assets_dir, 'blocks', 'blue_block.xml')
        if red_block_path is None:
            red_block_path = os.path.join(assets_dir, 'blocks', 'red_block.xml')

        # Load blue block
        self._blue_block_prop = composer.ModelWrapperEntity(
            mjcf.from_path(blue_block_path)
        )
        self._scene.add_free_entity(self._blue_block_prop)

        # Load red block
        self._red_block_prop = composer.ModelWrapperEntity(
            mjcf.from_path(red_block_path)
        )
        self._scene.add_free_entity(self._red_block_prop)

        # Remove freejoints to use PropPlacer
        for prop in [self._blue_block_prop, self._red_block_prop]:
            freejoint = traversal_utils.get_freejoint(
                prop.mjcf_model.find_all('body')[0]
            )
            if freejoint:
                freejoint.remove()

        # Add target sites
        self._scene.mjcf_model.worldbody.add(
            'site', 
            name='blue_target', 
            pos=f"{BLUE_TARGET_POS[0]} {BLUE_TARGET_POS[1]} {BLUE_TARGET_POS[2]}",
            size=f"{_TARGET_RADIUS} 0.001", 
            rgba="0 0 1 0.3", 
            type="cylinder"
        )
        self._scene.mjcf_model.worldbody.add(
            'site', 
            name='red_target', 
            pos=f"{RED_TARGET_POS[0]} {RED_TARGET_POS[1]} {RED_TARGET_POS[2]}",
            size=f"{_TARGET_RADIUS} 0.001", 
            rgba="1 0 0 0.3", 
            type="cylinder"
        )

        # Create prop placers for blocks
        self._placers = [
            initializers.PropPlacer(
                props=[self._blue_block_prop],
                position=blue_block_uniform_position,
                quaternion=block_z_rotation,
                ignore_collisions=True,
                settle_physics=False,
            ),
            initializers.PropPlacer(
                props=[self._red_block_prop],
                position=red_block_uniform_position,
                quaternion=block_z_rotation,
                ignore_collisions=True,
                settle_physics=True,  # Settle after last block
            ),
        ]

        # Update qpos for two blocks (each has 7 DOF: 3 pos + 4 quat)
        extra_qpos = np.zeros((14,))

        scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
        if scene_key is not None:
            scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for placer in self._placers:
            placer(physics, random_state)

    def get_reward(self, physics):
        """Returns reward based on blocks being in their target zones.
        
        Returns 0.5 for each block in its correct target, 1.0 for both blocks sorted correctly.
        """
        # Get positions of blocks
        blue_body = self._blue_block_prop.mjcf_model.find_all('body')[0]
        red_body = self._red_block_prop.mjcf_model.find_all('body')[0]
        blue_geom = self._blue_block_prop.mjcf_model.find_all('geom')[0]
        red_geom = self._red_block_prop.mjcf_model.find_all('geom')[0]

        blue_block_pos = physics.bind(blue_body).xpos
        red_block_pos = physics.bind(red_body).xpos
        
        # Check if blocks are in their target zones (horizontal distance only)
        blue_dist = np.linalg.norm(blue_block_pos[:2] - BLUE_TARGET_POS[:2])
        red_dist = np.linalg.norm(red_block_pos[:2] - RED_TARGET_POS[:2])
        
        blue_in_target = blue_dist < _TARGET_RADIUS
        red_in_target = red_dist < _TARGET_RADIUS

        # Require blocks to be resting on the ground plane.
        ground_z = 0.0
        ground_tol = 0.005  # 5mm tolerance for contact/settling
        blue_on_ground = abs(blue_block_pos[2] - (ground_z + blue_geom.size[2])) <= ground_tol
        red_on_ground = abs(red_block_pos[2] - (ground_z + red_geom.size[2])) <= ground_tol
        
        # Return reward
        reward = 0.0
        if blue_in_target and blue_on_ground:
            reward += 0.5
        if red_in_target and red_on_ground:
            reward += 0.5
        
        return reward
