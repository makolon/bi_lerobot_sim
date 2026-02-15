"""Creates lerobot sim task environments using dm_control.composer."""

import inspect
from bilerobot_sim.tasks import block_pnp
from bilerobot_sim.tasks import block_sort
from bilerobot_sim.tasks import block_stack

from dm_control import composer
import immutabledict
import numpy as np


DEFAULT_CAMERAS = (
    'overhead_cam',
    'front_cam',
    'left_wrist_cam',
    'right_wrist_cam',
)

DEFAULT_CONTROL_TIMESTEP = 0.02

TASK_FACTORIES = immutabledict.immutabledict({
    'BlockStack': (block_stack.BlockStack, {}),
    'BlockPnP': (block_pnp.BlockPnP, {}),
    'BlockSort': (block_sort.BlockSort, {}),
})


def create_task_env(
    task_name: str,
    time_limit: float,
    random_state: np.random.RandomState | int | None = None,
    control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    **kwargs,
) -> composer.Environment:
    """Creates an lerobot sim task environment.

    Args:
        task_name: The registered name of the task to create/
        time_limit: Time limit per episode in seconds.
        random_state: Random seed for the environment.
        control_timestep: Control timestep for the task.
        cameras: Tuple of camera names to use.
        **kwargs: Extra kwargs passed to the environment.

    Returns:
        A configured dm_control.composer.Environment.
    Raises:
        ValueError: If the task_name is not recognized.
    """
    if task_name not in TASK_FACTORIES:
        raise ValueError(
            f'Unknown task_name: {task_name}. Available tasks:'
            f' {list(TASK_FACTORIES.keys())}'
        )

    task_class, task_kwargs = TASK_FACTORIES[task_name]

    # remove any kwargs that are not in the task constructor
    signature = inspect.signature(task_class.__init__)
    task_class_kwargs = set(signature.parameters.keys())
    task_class_kwargs.remove('self')
    # adding allowed args in base classes
    task_class_kwargs.add('mjcf_root')
    kwargs = {k: v for k, v in kwargs.items() if k in task_class_kwargs}
    constructor_kwargs = {
        'control_timestep': control_timestep,
        'cameras': cameras,
        **task_kwargs,
    }
    kwargs.update(constructor_kwargs)

    task_instance = task_class(**kwargs)

    return composer.Environment(
        task_instance,
        random_state=random_state,
        time_limit=time_limit,
        strip_singleton_obs_buffer_dim=True,
        raise_exception_on_physics_error=False,
        delayed_observation_padding=composer.ObservationPadding.INITIAL_VALUE,
    )
