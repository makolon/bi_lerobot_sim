# Bimanual LeRobot Sim (SO101)

BiLeRobot Sim is a Python library that defines a dual-arm (bimanual) simulation
environment for the SO101 robot. It includes manipulation tasks for robot
learning and evaluation.

## Installation

Install with pip:

```bash
# create a virtual environment and pip install
pip install -e .
```

or run directly with uv:

```bash
pip install uv
uv run <script>.py
```

Tell MuJoCo which backend to use; otherwise the simulation may be slow:

```bash
export MUJOCO_GL='egl'
```

## Tasks

Currently available tasks (see `bilerobot_sim/task_suite.py`):

- `BlockStack` — stack three colored blocks (blue, yellow, red)

## Viewer / Video Recording

### Interactive viewer (requires `mjpython` on macOS)

```bash
python bilerobot_sim/viewer.py --task_name BlockStack
```

### Record overhead camera to MP4 (no viewer required)

```bash
python bilerobot_sim/example.py --task_name BlockStack --num_steps 500 --output_dir ./outputs
```

This will save `simulation.mp4` under the specified output directory.

## Tips

- If stepping is slow, verify `MUJOCO_GL` (e.g., `'egl'` on Linux servers).
- On macOS, the interactive viewer requires `mjpython`; if that fails, use the
	recording script instead.
- Task list and kwargs are in `bilerobot_sim/task_suite.py`.

## Note
This project is a derivative simulation for SO101 and is not an officially
supported Google product.
