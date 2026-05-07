# MECS6616 Spring 2026 Assignment 2: Imitation Learning

This repository contains the completed Assignment 2 notebook and saved artifacts for the Cheetah imitation-learning task. The project trains and evaluates MLP and CVAE policies on the `dm_control` Cheetah Run environment using the provided no-perturbation dataset.

## Contents

- `mecs6616_Spring2026_Assignment2.ipynb` - main assignment notebook with setup, model definitions, training, and evaluation cells.
- `dataset_no_perturbation.npz` - demonstration dataset used for policy training.
- `mlp_policy.pth` - trained MLP imitation policy checkpoint.
- `cvae_mlp_policy.pth` - trained CVAE imitation policy checkpoint.
- `act_normalizer.pt` - min-max state/action normalization statistics used by the CVAE policy.
- `models/` and `models_tune/` - saved training and tuning checkpoints.
- `scripts/generate_media.py` - local script for exporting rollout videos and plots from the saved CVAE policy.
- `downloads/` - generated rollout media exported from the project.

## Environment

The notebook was written for Colab-style execution and installs:

- `dm_control`
- `imageio`
- `imageio-ffmpeg`
- `torch`
- `matplotlib`
- `numpy`
- `tqdm`

For local execution in this workspace, use the included virtual environment:

```bash
./.venv/bin/python scripts/generate_media.py
```

On macOS, the media export script uses `MUJOCO_GL=glfw` so MuJoCo can render without the Linux-only OSMesa library used in the notebook setup cell.

## Running the Project

Open and run `mecs6616_Spring2026_Assignment2.ipynb` from top to bottom to reproduce the assignment workflow. The training cells skip retraining when the required checkpoint files already exist:

- `mlp_policy.pth`
- `cvae_mlp_policy.pth`

The final evaluation cell runs 100 independent Cheetah episodes and reports the average penalized reward and assignment score.

## Generated Media

Rollout media is stored in `downloads/`:

- `downloads/cvae_cheetah_rollout.mp4` - 4-second CVAE policy rollout video.
- `downloads/cvae_cheetah_first_frame.png` - first rendered frame from the rollout.
- `downloads/cvae_cheetah_rollout_log.png` - reward, position, and velocity plots from the rollout.

Regenerate these files with:

```bash
./.venv/bin/python scripts/generate_media.py
```

Optional arguments:

```bash
./.venv/bin/python scripts/generate_media.py --duration 4.0 --seed 42 --output-dir downloads
```

## Submission Artifacts

For the CourseWorks submission, include the notebook link and upload the trained checkpoint files:

- `mlp_policy.pth`
- `cvae_mlp_policy.pth`
