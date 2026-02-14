"""Block Pick and Place task."""

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
_BOWL_RESET_HEIGHT = 0.02

_BOWL_INNER_RADIUS_SCALE = 0.8
_BOWL_INNER_HEIGHT_SCALE = 0.7
_INSIDE_MARGIN = 0.0015
_RIM_CLEARANCE = 0.001

# Blue block position (start)
blue_block_uniform_position = distributions.Uniform(
    low=[-0.04, 0.08, _BLOCK_RESET_HEIGHT],
    high=[0.04, 0.16, _BLOCK_RESET_HEIGHT],
    single_sample=True,
)

# Bowl position (target)
_BOWL_TARGET_POS = np.array([0.45, 0.12, _BOWL_RESET_HEIGHT])
bowl_uniform_position = distributions.Uniform(
    low=_BOWL_TARGET_POS.tolist(),
    high=_BOWL_TARGET_POS.tolist(),
    single_sample=True,
)

# Random rotation around Z-axis
block_z_rotation = rotations.UniformQuaternion()


class BlockPnP(lerobot_task.LeRobotTask):
    """Pick and place a blue block into a bowl."""

    def __init__(
        self,
        blue_block_path: str | None = None,
        bowl_path: str | None = None,
        bowl_inner_radius: float | None = None,
        bowl_inner_height: float | None = None,
        **kwargs,
    ):
        """Initializes a new `BlockPnP` task.

        Args:
            blue_block_path: Path to asset of the blue block.
            bowl_path: Path to asset of the bowl.
            bowl_inner_radius: Optional inner radius of the bowl (meters).
            bowl_inner_height: Optional inner height of the bowl (meters).
            **kwargs: Additional args to pass to the base class.
        """
        super().__init__(**kwargs)

        assets_dir = os.path.join(os.path.dirname(__file__), '../assets')

        # Default block path if not provided
        if blue_block_path is None:
            blue_block_path = os.path.join(assets_dir, 'blocks', 'blue_block.xml')
        if bowl_path is None:
            bowl_path = os.path.join(assets_dir, 'bowl', 'model.xml')

        self._bowl_inner_radius = None
        self._bowl_inner_height = None
        self._bowl_inner_center_local = None
        self._block_radius = None

        # Load blue block
        self._blue_block_prop = composer.ModelWrapperEntity(
            mjcf.from_path(blue_block_path)
        )
        self._bowl_prop = composer.ModelWrapperEntity(mjcf.from_path(bowl_path))
        self._scene.add_free_entity(self._blue_block_prop)
        self._scene.add_free_entity(self._bowl_prop)

        # Remove freejoint to use PropPlacer
        for prop in [self._blue_block_prop, self._bowl_prop]:
            freejoint = traversal_utils.get_freejoint(
                prop.mjcf_model.find_all('body')[0]
            )
            if freejoint:
                freejoint.remove()

        self._placers = [
            initializers.PropPlacer(
                props=[self._bowl_prop],
                position=copy.deepcopy(bowl_uniform_position),
                quaternion=deterministic.Identity(),
                ignore_collisions=True,
                settle_physics=False,
                max_attempts_per_prop=100,
            ),
            initializers.PropPlacer(
                props=[self._blue_block_prop],
                position=copy.deepcopy(blue_block_uniform_position),
                quaternion=block_z_rotation,
                ignore_collisions=True,
                settle_physics=False,
                max_attempts_per_prop=100,
            ),
            initializers.PropPlacer(
                props=[self._blue_block_prop, self._bowl_prop],
                position=deterministic.Identity(),
                quaternion=deterministic.Identity(),
                ignore_collisions=True,
                settle_physics=True,
            ),
        ]

        if bowl_inner_radius is not None:
            self._bowl_inner_radius = bowl_inner_radius
        if bowl_inner_height is not None:
            self._bowl_inner_height = bowl_inner_height

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for placer in self._placers:
            placer(physics, random_state)

    def _select_largest_geom(self, physics, prop):
        geoms = prop.mjcf_model.find_all('geom')
        if len(geoms) == 1:
            return geoms[0]
        largest_geom = geoms[0]
        largest_rbound = -1.0
        for geom in geoms:
            rbound = float(physics.bind(geom).rbound)
            if rbound > largest_rbound:
                largest_rbound = rbound
                largest_geom = geom
        return largest_geom

    def _infer_radius_height_from_geom(self, physics, geom):
        geom_type = getattr(geom, 'type', None) or 'mesh'
        size = np.array(geom.size, dtype=np.float32) if geom.size is not None else None
        center_local = (
            np.array(geom.pos, dtype=np.float32)
            if geom.pos is not None
            else np.zeros((3,), dtype=np.float32)
        )

        if geom_type == 'sphere' and size is not None:
            radius = float(size[0])
            height = float(size[0] * 2.0)
        elif geom_type == 'cylinder' and size is not None:
            radius = float(size[0])
            height = float(size[1] * 2.0)
        elif geom_type == 'capsule' and size is not None:
            radius = float(size[0])
            height = float(size[1] * 2.0 + radius * 2.0)
        elif geom_type == 'box' and size is not None:
            radius = float(np.sqrt(size[0] ** 2 + size[1] ** 2))
            height = float(size[2] * 2.0)
        else:
            rbound = float(physics.bind(geom).rbound)
            radius = rbound
            height = rbound * 2.0
        return radius, height, center_local

    def _ensure_inferred_sizes(self, physics):
        if self._block_radius is None:
            block_geom = self._select_largest_geom(physics, self._blue_block_prop)
            radius, _, _ = self._infer_radius_height_from_geom(physics, block_geom)
            self._block_radius = radius

        if (
            self._bowl_inner_radius is None
            or self._bowl_inner_height is None
            or self._bowl_inner_center_local is None
        ):
            bowl_geom = self._select_largest_geom(physics, self._bowl_prop)
            radius, height, center_local = self._infer_radius_height_from_geom(
                physics, bowl_geom
            )
            if self._bowl_inner_radius is None:
                self._bowl_inner_radius = radius * _BOWL_INNER_RADIUS_SCALE
            if self._bowl_inner_height is None:
                self._bowl_inner_height = height * _BOWL_INNER_HEIGHT_SCALE
            if self._bowl_inner_center_local is None:
                self._bowl_inner_center_local = center_local

    def get_reward(self, physics):
        """Returns 1.0 if the blue block is inside the bowl; otherwise 0.0."""
        self._ensure_inferred_sizes(physics)

        block_body = self._blue_block_prop.mjcf_model.find_all('body')[0]
        bowl_body = self._bowl_prop.mjcf_model.find_all('body')[0]

        block_pos = physics.bind(block_body).xpos
        bowl_bind = physics.bind(bowl_body)
        bowl_pos = bowl_bind.xpos
        bowl_xmat = bowl_bind.xmat.reshape(3, 3)

        rel_pos = bowl_xmat.T @ (block_pos - bowl_pos)
        rel_pos = rel_pos - self._bowl_inner_center_local

        radial_dist = float(np.linalg.norm(rel_pos[:2]))
        z_pos = float(rel_pos[2])

        inner_radius = float(self._bowl_inner_radius)
        inner_height = float(self._bowl_inner_height)
        block_radius = float(self._block_radius)

        radial_ok = radial_dist <= (inner_radius - block_radius - _INSIDE_MARGIN)
        z_min = -inner_height / 2.0 + block_radius - _INSIDE_MARGIN
        z_max = inner_height / 2.0 - block_radius - _RIM_CLEARANCE
        z_ok = (z_pos >= z_min) and (z_pos <= z_max)

        return 1.0 if (radial_ok and z_ok) else 0.0
