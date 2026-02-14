import time
from typing import Sequence
from absl import app
from absl import flags
from absl import logging
from bi_lerobot_sim import task_suite
from dm_control import composer
import numpy as np
import cv2
from pathlib import Path


_TASK_NAME = flags.DEFINE_enum(
    'task_name',
    'BlockStack',
    task_suite.TASK_FACTORIES.keys(),
    'Task name.',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    './outputs',
    'Output directory for MP4 videos.',
)

_NUM_STEPS = flags.DEFINE_integer(
    'num_steps',
    500,
    'Number of simulation steps to record.',
)

_LOG_STEPS = 100
_DT = 0.02
_IMAGE_SIZE = (480, 640)
_LEROBOT_CAMERAS = {
    'overhead_cam': _IMAGE_SIZE,
    'front_cam': _IMAGE_SIZE,
    'left_wrist_cam': _IMAGE_SIZE,
    'right_wrist_cam': _IMAGE_SIZE,
}
_REST_ACTION = np.asarray([
    0.0, -1.70, 1.70, 0.921, 0.0120, 0.0,
    0.0, -1.70, 1.70, 0.921, 0.0120, 0.0,
], dtype=np.float32)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 2:
        raise app.UsageError('Too many command-line arguments.')

    # Setup output directory
    output_dir = Path(_OUTPUT_DIR.value)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info('Output directory: %s', output_dir)

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
        update_interval=1, 
        **kwargs
    )
    env = composer.Environment(
        task=task,
        time_limit=float('inf'),
        random_state=np.random.RandomState(0),
        recompile_mjcf_every_episode=False,
        strip_singleton_obs_buffer_dim=True,
        delayed_observation_padding=composer.ObservationPadding.INITIAL_VALUE,
    )
    
    current_action = _REST_ACTION.copy()
    
    # Create video writer
    video_path = output_dir / 'simulation.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(1.0 / _DT)
    video_writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (_IMAGE_SIZE[1], _IMAGE_SIZE[0])
    )
    
    if not video_writer.isOpened():
        logging.error('Failed to open video writer for %s', video_path)
        env.close()
        return
    
    logging.info('Recording %d steps to %s at %d fps', _NUM_STEPS.value, video_path, fps)
    
    # Reset environment
    timestep = env.reset()
    
    total_reward = 0.0
    time_stepping = 0
    
    # Record simulation steps
    for step in range(_NUM_STEPS.value):
        frame_start_time = time.time()
        
        # Step simulation
        timestep = env.step(current_action)
        total_reward += float(timestep.reward or 0.0)
        
        step_end_time = time.time()
        time_stepping += step_end_time - frame_start_time

        # OpenCV uses BGR format, but renderer returns RGB
        pixels = timestep.observation["front_cam"]
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
        
        if (step + 1) % _LOG_STEPS == 0:
            logging.info(
                'Step: %d/%d, Total Reward: %.4f',
                step + 1, _NUM_STEPS.value, total_reward
            )
            logging.info(
                'Stepping time per step:\t%.4fs, total:\t%.2fs',
                time_stepping / _LOG_STEPS,
                time_stepping,
            )
            time_stepping = 0
    
    # Close video writer
    video_writer.release()
    logging.info(
        'Recording completed. Total steps: %d, Total reward: %.4f',
        _NUM_STEPS.value, total_reward
    )
    logging.info('Video saved to %s', video_path)
    
    env.close()
    logging.info('All episodes recorded successfully.')



if __name__ == '__main__':
    app.run(main)
