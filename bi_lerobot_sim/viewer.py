"""Interactive viewer for running model inference.

Instructions:

- I = enter new instruction.
- space bar = pause/restart.
- backspace = reset environment.
- mouse right moves the camera
- mouse left rotates the camera
- double click to select an object

When the environment is not running:
- ctrl + mouse left rotates a selected object
- ctrl + mouse right moves a selected object

When the environment is running:
- ctrl + mouse left applies torque to an object
- ctrl + mouse right applies force to an object
"""

import time
from typing import Sequence
from absl import app
from absl import flags
from absl import logging
from bi_lerobot_sim import task_suite
from dm_control import composer
import dm_env
import mujoco
import mujoco.viewer
import numpy as np


_TASK_NAME = flags.DEFINE_enum(
    'task_name',
    'BlockStack',
    task_suite.TASK_FACTORIES.keys(),
    'Task name.',
)

# --- Global State for Viewer Interaction ---
_GLOBAL_STATE = {
    '_IS_RUNNING': True,
    '_SHOULD_RESET': False,
    '_SINGLE_STEP': False,
}
_LOG_STEPS = 100
_DT = 0.02
_IMAGE_SIZE = (480, 848)
_LEROBOT_CAMERAS = {
    'overhead_cam': _IMAGE_SIZE,
    'front_cam': _IMAGE_SIZE,
    'left_wrist_cam': _IMAGE_SIZE,
    'right_wrist_cam': _IMAGE_SIZE,
}
_INIT_ACTION = np.asarray([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
], dtype=np.float32)


def _key_callback(key: int) -> None:
    """Viewer callbacks for key-presses."""
    if key == 32:  # Space bar
        _GLOBAL_STATE['_IS_RUNNING'] = not _GLOBAL_STATE['_IS_RUNNING']
        logging.info('RUNNING = %s', _GLOBAL_STATE['_IS_RUNNING'])
    elif key == 259:  # Backspace
        _GLOBAL_STATE['_SHOULD_RESET'] = True
        logging.info('RESET = %s', _GLOBAL_STATE['_SHOULD_RESET'])
    elif key == 262:  # Right arrow
        _GLOBAL_STATE['_SINGLE_STEP'] = True
        _GLOBAL_STATE['_IS_RUNNING'] = True  # Allow one step to proceed
        logging.info('_SINGLE_STEP = %s', _GLOBAL_STATE['_SINGLE_STEP'])
    else:
        logging.info('UNKNOWN KEY PRESS = %s', key)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 2:
        raise app.UsageError('Too many command-line arguments.')

    logging.info('Initializing %s environment...', _TASK_NAME.value)
    if _TASK_NAME.value not in task_suite.TASK_FACTORIES.keys():
        raise ValueError(
            f'Unknown task_name: {_TASK_NAME.value}. Available tasks:'
            f' {list(task_suite.TASK_FACTORIES.keys())}'
        )
    task_class, kwargs = task_suite.TASK_FACTORIES[_TASK_NAME.value]
    task = task_class(
        cameras=tuple(_LEROBOT_CAMERAS.keys()), 
        control_timestep=_DT, 
        update_interval=25, 
        **kwargs
    )
    env = composer.Environment(
        task=task,
        time_limit=float('inf'),  # No explicit time limit from the environment
        random_state=np.random.RandomState(0),  # For reproducibility
        recompile_mjcf_every_episode=False,
        strip_singleton_obs_buffer_dim=True,
        delayed_observation_padding=composer.ObservationPadding.INITIAL_VALUE,
    )
    env.reset()

    viewer_model = env.physics.model.ptr
    viewer_data = env.physics.data.ptr
    
    # Get action spec for manual control
    current_action = _INIT_ACTION.copy()

    logging.info('Launching viewer...')
    with mujoco.viewer.launch_passive(
        viewer_model, viewer_data, key_callback=_key_callback
    ) as viewer_handle:
        viewer_handle.sync()
        logging.info(
            'Viewer started. Press Space to play/pause, Backspace to reset.'
        )
        logging.info(
            'Using manual control with zero action (gravity only).'
        )
        
        while viewer_handle.is_running():
            timestep = env.reset()
            viewer_handle.sync()

            steps = 0
            time_stepping = 0
            sync_time = 0

            while not timestep.last():
                steps += 1
                
                if _GLOBAL_STATE['_IS_RUNNING'] or _GLOBAL_STATE['_SINGLE_STEP']:
                    frame_start_time = time.time()
                    
                    # Use zero action (no control, just physics simulation)
                    action = current_action
                    
                    current_timestep = env.step(action)
                    step_end_time = time.time()
                    time_stepping += step_end_time - frame_start_time

                    viewer_handle.sync()
                    sync_time += time.time() - step_end_time

                    if steps % _LOG_STEPS == 0:
                        logging.info('Step: %s', steps)
                        logging.info(
                            'Stepping time per step:\t%ss, total:\t%ss',
                            time_stepping / _LOG_STEPS,
                            time_stepping,
                        )
                        logging.info(
                            'Sync time per step:\t%ss, total:\t%ss',
                            sync_time / _LOG_STEPS,
                            sync_time,
                        )
                        time_stepping = 0
                        sync_time = 0

                    if _GLOBAL_STATE['_SHOULD_RESET']:
                        # Reset was pressed mid-episode
                        _GLOBAL_STATE['_SHOULD_RESET'] = False
                        current_timestep = current_timestep._replace(
                            step_type=dm_env.StepType.LAST
                        )

                    assert (
                        not current_timestep.first()
                    ), 'Environment auto-reseted mid-episode unexpectedly.'
                    timestep = current_timestep

                    if _GLOBAL_STATE['_SINGLE_STEP']:
                        _GLOBAL_STATE['_SINGLE_STEP'] = False
                        _GLOBAL_STATE['_IS_RUNNING'] = False  # Pause after single step

                with viewer_handle.lock():
                    # Apply perturbations if active (e.g. mouse drag)
                    if viewer_handle.perturb.active:
                        if _GLOBAL_STATE['_IS_RUNNING']:
                            mujoco.mjv_applyPerturbForce(
                                viewer_model,
                                viewer_data,
                                viewer_handle.perturb,
                            )
                        else:
                            mujoco.mjv_applyPerturbPose(
                                viewer_model,
                                viewer_data,
                                viewer_handle.perturb,
                                flg_paused=1,
                            )
                            mujoco.mj_kinematics(viewer_model, viewer_data)
                    viewer_handle.sync()

                if not _GLOBAL_STATE['_IS_RUNNING']:
                    time.sleep(0.01)  # Yield to other threads if paused

            if _GLOBAL_STATE[
                '_SHOULD_RESET'
            ]:  # Reset pressed at the very end of an episode
                _GLOBAL_STATE['_SHOULD_RESET'] = False
                
    logging.info('Viewer exited.')
    env.close()


if __name__ == '__main__':
    app.run(main)
