"""LeRobot SO ARM100 bimanual base task backed by MuJoCo MJCF."""

import collections
from collections.abc import Mapping
import copy
import dataclasses
import enum
import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer import initializers
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import variation_broadcaster
from dm_env import specs
import immutabledict
import numpy as np
from numpy import typing as npt


# Per-arm joint order: [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw].
_ARM_HOME: npt.NDArray[float] = np.array([0.0, -1.57, 1.57, 1.57, 0.0, 0.0])
_HOME_STATE_BY_ARM = np.concatenate([_ARM_HOME, _ARM_HOME])
HOME_CTRL: npt.NDArray[float] = _HOME_STATE_BY_ARM.astype(np.float32)
HOME_QPOS: npt.NDArray[float] = _HOME_STATE_BY_ARM.astype(np.float32)
HOME_CTRL.setflags(write=False)
HOME_QPOS.setflags(write=False)


SIM_GRIPPER_QPOS_OPEN: float = 1.75
SIM_GRIPPER_QPOS_CLOSE: float = -0.174
SIM_GRIPPER_CTRL_OPEN: float = 1.75
SIM_GRIPPER_CTRL_CLOSE: float = -0.174
FOLLOWER_GRIPPER_OPEN: float = 1.5155
FOLLOWER_GRIPPER_CLOSE: float = -0.06135
LEADER_GRIPPER_OPEN: float = 0.78
LEADER_GRIPPER_CLOSE: float = -0.04


@dataclasses.dataclass(frozen=True)
class GripperLimit:
    open: float
    close: float


GRIPPER_LIMITS = immutabledict.immutabledict({
    'sim_qpos': GripperLimit(
        open=SIM_GRIPPER_QPOS_OPEN,
        close=SIM_GRIPPER_QPOS_CLOSE,
    ),
    'sim_ctrl': GripperLimit(
        open=SIM_GRIPPER_CTRL_OPEN,
        close=SIM_GRIPPER_CTRL_CLOSE,
    ),
    'follower': GripperLimit(
        open=FOLLOWER_GRIPPER_OPEN,
        close=FOLLOWER_GRIPPER_CLOSE,
    ),
    'leader': GripperLimit(
        open=LEADER_GRIPPER_OPEN,
        close=LEADER_GRIPPER_CLOSE,
    ),
})


_DEFAULT_PHYSICS_DELAY_SECS: float = 0.3
_DEFAULT_JOINT_OBSERVATION_DELAY_SECS: float = 0.1

_ARM_NAMES: tuple[str, ...] = ('left', 'right')
_ARM_DOF: int = 6
_ARM_GRIPPER_INDEX: int = 5

_ARM_JOINTS_BY_SIDE: dict[str, tuple[str, ...]] = {
    arm: (
        f'{arm}_Rotation',
        f'{arm}_Pitch',
        f'{arm}_Elbow',
        f'{arm}_Wrist_Pitch',
        f'{arm}_Wrist_Roll',
        f'{arm}_Jaw',
    )
    for arm in _ARM_NAMES
}
_ALL_JOINTS: tuple[str, ...] = tuple(
    joint_name
    for arm in _ARM_NAMES
    for joint_name in _ARM_JOINTS_BY_SIDE[arm]
)


class GeomGroup(enum.IntFlag):
    NONE = 0
    ARM = enum.auto()
    GRIPPER = enum.auto()
    TABLE = enum.auto()
    OBJECT = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()


class LeRobotTask(composer.Task):
    """The base SO ARM100 task for bimanual manipulation."""

    def __init__(
        self,
        control_timestep: float,
        cameras: tuple[str, ...] = (
            'overhead_cam',
            'front_cam',
            'left_wrist_cam',
            'right_wrist_cam',
        ),
        camera_resolution: tuple[int, int] = (480, 640),
        joints_observation_delay_secs: (
            variation.Variation | float
        ) = _DEFAULT_JOINT_OBSERVATION_DELAY_SECS,
        image_observation_enabled: bool = True,
        image_observation_delay_secs: (
            variation.Variation | float
        ) = _DEFAULT_PHYSICS_DELAY_SECS,
        update_interval: int = 1,
        waist_joint_limit: float = np.pi / 2,
        terminate_episode=True,
        mjcf_root: str | None = None,
    ):
        self._waist_joint_limit = waist_joint_limit
        self._terminate_episode = terminate_episode

        self._scene = Arena(
            camera_resolution=camera_resolution,
            mjcf_root_path=mjcf_root,
        )
        self._scene.mjcf_model.option.flag.multiccd = 'enable'
        self._scene.mjcf_model.option.noslip_iterations = 0

        self.control_timestep = control_timestep

        self._joints = [
            self._scene.mjcf_model.find('joint', name) for name in _ALL_JOINTS
        ]

        obs_dict = collections.OrderedDict()

        shared_delay = variation_broadcaster.VariationBroadcaster(
            image_observation_delay_secs / self.physics_timestep
        )
        cameras_entities = [
            self.root_entity.mjcf_model.find('camera', name) for name in cameras
        ]
        for camera_entity in cameras_entities:
            if camera_entity is None:
                raise ValueError(f'Unknown camera in scene: {cameras}')
            obs_dict[camera_entity.name] = observable.MJCFCamera(
                camera_entity,
                height=camera_resolution[0],
                width=camera_resolution[1],
                update_interval=update_interval,
                buffer_size=1,
                delay=shared_delay.get_proxy(),
                aggregator=None,
                corruptor=None,
            )
            obs_dict[camera_entity.name].enabled = True

        lerobot_observables = LeRobotObservables(
            self.root_entity,
        )
        lerobot_observables.enable_all()
        obs_dict.update(lerobot_observables.as_dict())
        self._task_observables = obs_dict

        if joints_observation_delay_secs:
            self._task_observables['undelayed_joints_pos'] = copy.copy(
                self._task_observables['joints_pos']
            )
            self._task_observables['undelayed_joints_vel'] = copy.copy(
                self._task_observables['joints_vel']
            )
            self._task_observables['joints_pos'].configure(
                delay=joints_observation_delay_secs / self.physics_timestep
            )
            self._task_observables['joints_vel'].configure(
                delay=joints_observation_delay_secs / self.physics_timestep
            )
            self._task_observables['delayed_joints_pos'] = copy.copy(
                self._task_observables['joints_pos']
            )
            self._task_observables['delayed_joints_vel'] = copy.copy(
                self._task_observables['joints_vel']
            )

        self._task_observables['physics_state'].enabled = (
            image_observation_enabled
        )
        if image_observation_delay_secs:
            self._task_observables['delayed_physics_state'] = copy.copy(
                self._task_observables['physics_state']
            )
            self._task_observables['delayed_physics_state'].configure(
                delay=shared_delay.get_proxy(),
            )

        self._all_props = []
        self._all_prop_placers = []
        if self._all_prop_placers:
            self._all_prop_placers.append(
                initializers.PropPlacer(
                    props=self._all_props,
                    position=deterministic.Identity(),
                    ignore_collisions=True,
                    settle_physics=True,
                )
            )

    @property
    def root_entity(self) -> composer.Entity:
        return self._scene

    @property
    def task_observables(self) -> Mapping[str, observable.Observable]:
        return dict(**self._task_observables)

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        minimum = physics.model.actuator_ctrlrange[:, 0].astype(np.float32)
        maximum = physics.model.actuator_ctrlrange[:, 1].astype(np.float32)

        for arm_idx in range(len(_ARM_NAMES)):
            offset = arm_idx * _ARM_DOF
            minimum[offset] = -self._waist_joint_limit
            maximum[offset] = self._waist_joint_limit
            minimum[offset + _ARM_GRIPPER_INDEX] = GRIPPER_LIMITS['follower'].close
            maximum[offset + _ARM_GRIPPER_INDEX] = GRIPPER_LIMITS['follower'].open

        return specs.BoundedArray(
            shape=(len(_ALL_JOINTS),),
            dtype=np.float32,
            minimum=minimum,
            maximum=maximum,
        )

    @classmethod
    def convert_gripper(
        cls,
        gripper_value: npt.NDArray[float],
        from_name: str,
        to_name: str,
    ) -> npt.NDArray[float]:
        from_limits = GRIPPER_LIMITS[from_name]
        to_limits = GRIPPER_LIMITS[to_name]
        return (gripper_value - from_limits.close) / (
            from_limits.open - from_limits.close
        ) * (to_limits.open - to_limits.close) + to_limits.close

    def before_step(
        self,
        physics: mjcf.Physics,
        action: npt.ArrayLike,
        random_state: np.random.RandomState,
    ) -> None:
        del random_state
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (len(_ALL_JOINTS),):
            raise ValueError(
                f'Expected bimanual action shape {(len(_ALL_JOINTS),)}, got {action.shape}'
            )

        ctrl = physics.data.ctrl
        for arm_idx in range(len(_ARM_NAMES)):
            offset = arm_idx * _ARM_DOF
            ctrl[offset: offset + (_ARM_DOF - 1)] = action[offset: offset + (_ARM_DOF - 1)]
            gripper_cmd = action[offset + _ARM_GRIPPER_INDEX]
            gripper_ctrl = LeRobotTask.convert_gripper(
                np.array([gripper_cmd], dtype=np.float32),
                'follower',
                'sim_ctrl',
            )
            ctrl[offset + _ARM_GRIPPER_INDEX] = gripper_ctrl[0]

    def get_reward(self, physics: mjcf.Physics) -> float:
        return 0.0

    def get_discount(self, physics: mjcf.Physics) -> float:
        if self.should_terminate_episode(physics):
            return 0.0
        return 1.0

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        if self._terminate_episode:
            reward = self.get_reward(physics)
            if reward >= 1.0:
                return True
        return False

    def initialize_episode(
        self,
        physics: mjcf.Physics,
        random_state: np.random.RandomState,
    ) -> None:
        arm_joints_bound = physics.bind(self._joints)
        arm_joints_bound.qpos[:] = HOME_QPOS
        np.copyto(physics.data.ctrl, HOME_CTRL)

        for prop_placer in self._all_prop_placers:
            prop_placer(physics, random_state)


class LeRobotObservables(composer.Observables):
    """LeRobot bimanual observables."""

    def as_dict(
        self,
        fully_qualified: bool = False,
    ) -> collections.OrderedDict[str, observable.Observable]:
        return super().as_dict(fully_qualified=fully_qualified)

    @define.observable
    def joints_pos(self) -> observable.Observable:
        def _get_joints_pos(physics):
            joints_qpos = np.asarray(physics.bind([
                self._entity.mjcf_model.find('joint', name)
                for name in _ALL_JOINTS
            ]).qpos, dtype=np.float32).copy()
            for arm_idx in range(len(_ARM_NAMES)):
                gripper_index = arm_idx * _ARM_DOF + _ARM_GRIPPER_INDEX
                joints_qpos[gripper_index] = LeRobotTask.convert_gripper(
                    np.array([joints_qpos[gripper_index]], dtype=np.float32),
                    'sim_qpos',
                    'follower',
                )[0]
            return joints_qpos

        return observable.Generic(_get_joints_pos)

    @define.observable
    def commanded_joints_pos(self) -> observable.Observable:
        def _get_joints_cmd(physics):
            joints_cmd = np.asarray(physics.data.ctrl, dtype=np.float32).copy()
            for arm_idx in range(len(_ARM_NAMES)):
                gripper_index = arm_idx * _ARM_DOF + _ARM_GRIPPER_INDEX
                joints_cmd[gripper_index] = LeRobotTask.convert_gripper(
                    np.array([joints_cmd[gripper_index]], dtype=np.float32),
                    'sim_ctrl',
                    'follower',
                )[0]
            return joints_cmd

        return observable.Generic(_get_joints_cmd)

    @define.observable
    def joints_vel(self) -> observable.Observable:
        return observable.MJCFFeature(
            'qvel',
            [self._entity.mjcf_model.find('joint', name) for name in _ALL_JOINTS],
        )

    @define.observable
    def physics_state(self) -> observable.Observable:
        return observable.Generic(lambda physics: physics.get_state())


class Arena(composer.Arena):
    """Standard arena for SO ARM100 bimanual setup."""

    def __init__(
        self,
        *args,
        camera_resolution,
        mjcf_root_path: str | None = None,
        **kwargs,
    ):
        self._camera_resolution = camera_resolution
        self.textures = []
        self._mjcf_root_path = mjcf_root_path
        super().__init__(*args, **kwargs)

    def _build(self, name: str | None = None) -> None:
        del name
        if not self._mjcf_root_path:
            self._mjcf_root_path = os.path.join(
                os.path.dirname(__file__),
                '../assets',
                'so_arm100/scene.xml',
            )

        self._mjcf_root = mjcf.from_path(
            path=self._mjcf_root_path,
            escape_separators=True,
        )
        self._mjcf_root.visual.__getattr__('global').offheight = (
            self._camera_resolution[0]
        )
        self._mjcf_root.visual.__getattr__('global').offwidth = (
            self._camera_resolution[1]
        )
