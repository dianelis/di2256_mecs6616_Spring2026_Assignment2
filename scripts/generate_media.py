"""Generate rollout media for the trained CVAE imitation policy."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from dm_control import suite

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MPC_ENV_GT:
    def __init__(self, env):
        self.env = env
        self.action_spec = env.action_spec()
        self.MPC_frequency = 25
        self.control_timestep = 1 / self.MPC_frequency
        self.MPC_repeats = int(self.control_timestep / env.control_timestep())
        self.clear()

    def clear(self):
        self.frames = []
        self.qposs = []
        self.qvels = []

    def reset(self):
        self.env.reset()

    def get_state(self):
        return self.env.physics.get_state()

    def step(self, action, record=False):
        reward = 0.0
        for _ in range(self.MPC_repeats):
            time_step = self.env.step(action)
            reward += time_step.reward
        if record:
            self.frames.append(self.env.physics.render(camera_id=0, width=256, height=256))
            self.qposs.append(copy.deepcopy(self.env.physics.data.qpos))
            self.qvels.append(copy.deepcopy(self.env.physics.data.qvel))
        return reward

    def get_log(self):
        return self.frames, self.qposs, self.qvels

    def close(self):
        self.env.close()


class MiniMaxStateActionNormalizer:
    def __init__(self, normalization_range=(-1, 1)):
        self.normalization_range = normalization_range
        self.state_min = None
        self.state_max = None
        self.action_min = None
        self.action_max = None
        self.next_state_min = None
        self.next_state_max = None

    def normalize_state(self, state):
        state_01 = (state - self.state_min) / (self.state_max - self.state_min)
        range_min, range_max = self.normalization_range
        return state_01 * (range_max - range_min) + range_min

    def denormalize_action(self, action_norm):
        range_min, range_max = self.normalization_range
        action_01 = (action_norm - range_min) / (range_max - range_min)
        return action_01 * (self.action_max - self.action_min) + self.action_min

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.normalization_range = checkpoint["normalization_range"]
        self.state_min = checkpoint["state_min"]
        self.state_max = checkpoint["state_max"]
        self.action_min = checkpoint["action_min"]
        self.action_max = checkpoint["action_max"]
        self.next_state_min = checkpoint["next_state_min"]
        self.next_state_max = checkpoint["next_state_max"]


def build_mlp(input_dim, hidden_dims, output_dim, dropout=0.1):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, dim_list, state_dim, state_horizon, action_dim, prediction_horizon, latent_dim):
        super().__init__()
        input_dim = state_horizon * state_dim + prediction_horizon * action_dim
        hidden_dims = dim_list[:-1] if len(dim_list) > 1 else []
        hidden_dim = dim_list[-1]
        self.backbone = build_mlp(input_dim, hidden_dims, hidden_dim, dropout=0.1)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs, action):
        hidden = self.backbone(torch.cat([obs, action], dim=1))
        return self.mu_head(hidden), self.logvar_head(hidden)


class Decoder(nn.Module):
    def __init__(self, dim_list, state_dim, state_horizon, action_dim, prediction_horizon, latent_dim):
        super().__init__()
        input_dim = state_horizon * state_dim + latent_dim
        output_dim = prediction_horizon * action_dim
        self.model = build_mlp(input_dim, dim_list, output_dim, dropout=0.1)

    def forward(self, obs, z):
        return self.model(torch.cat([obs, z], dim=1))


class CVAE(nn.Module):
    def __init__(self, encoder_dim_list, decoder_dim_list, state_dim, state_horizon, action_dim, prediction_horizon, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(encoder_dim_list, state_dim, state_horizon, action_dim, prediction_horizon, latent_dim)
        self.decoder = Decoder(decoder_dim_list, state_dim, state_horizon, action_dim, prediction_horizon, latent_dim)

    def sample_action(self, obs):
        z = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device, dtype=obs.dtype)
        return self.decoder(obs, z)


class CVAEAgent:
    def __init__(
        self,
        normalizer_path,
        model_path,
        state_dim=18,
        state_horizon=1,
        action_dim=6,
        prediction_horizon=6,
        execution_horizon=1,
        latent_dim=8,
    ):
        self.state_dim = state_dim
        self.state_horizon = state_horizon
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon
        self.execution_horizon = execution_horizon
        self.device = torch.device("cpu")
        self.normalizer = MiniMaxStateActionNormalizer()
        self.normalizer.load(normalizer_path)
        self.model = CVAE(
            encoder_dim_list=[512, 512, 256],
            decoder_dim_list=[512, 512, 256],
            state_dim=state_dim,
            state_horizon=state_horizon,
            action_dim=action_dim,
            prediction_horizon=prediction_horizon,
            latent_dim=latent_dim,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def plan(self, current_state):
        current_state = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
        state_norm = self.normalizer.normalize_state(current_state).unsqueeze(0)
        with torch.no_grad():
            action_norm = self.model.sample_action(state_norm.flatten(start_dim=1))
        action_norm = action_norm.view(1, self.prediction_horizon, self.action_dim).squeeze(0)
        return self.normalizer.denormalize_action(action_norm).numpy()


def save_log_plot(path, ticks, rewards, qposs, qvels):
    ticks = [t * 4 for t in ticks]
    qposs_array = np.asarray(qposs)
    qvels_array = np.asarray(qvels)
    fig, ax = plt.subplots(3, 3, sharex=True, figsize=(12, 8))

    ax[0, 0].plot(ticks, rewards)
    ax[0, 0].set_ylabel("reward")
    ax[0, 0].set_title("Reward")
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")

    labels = [("x position", "X Position"), ("z position", "Z Position"), ("pitch position", "Pitch Position")]
    for idx, (ylabel, title) in enumerate(labels):
        ax[1, idx].plot(ticks, qposs_array[:, idx])
        ax[1, idx].set_ylabel(ylabel)
        ax[1, idx].set_title(title)

    labels = [("x velocity", "X Velocity"), ("z velocity", "Z Velocity"), ("pitch velocity", "Pitch Velocity")]
    for idx, (ylabel, title) in enumerate(labels):
        ax[2, idx].plot(ticks, qvels_array[:, idx])
        ax[2, idx].set_ylabel(ylabel)
        ax[2, idx].set_xlabel("time")
        ax[2, idx].set_title(title)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_rollout(duration, seed):
    random_state = np.random.RandomState(seed)
    np.random.seed(seed)
    env = MPC_ENV_GT(suite.load("cheetah", "run", task_kwargs={"random": random_state}))
    agent = CVAEAgent("act_normalizer.pt", "cvae_mlp_policy.pth", prediction_horizon=6, execution_horizon=1, latent_dim=8)
    env.reset()
    num_steps = int(duration / env.control_timestep)
    rewards = []
    ticks = []
    action_sequence = []

    try:
        for step in range(num_steps):
            current_state = env.get_state()
            if len(action_sequence) == 0:
                action_sequence = agent.plan(current_state)[: agent.execution_horizon]
            action, action_sequence = action_sequence[0], action_sequence[1:]
            rewards.append(env.step(action, record=True))
            ticks.append(step)
        frames, qposs, qvels = env.get_log()
    finally:
        env.close()

    return frames, rewards, ticks, qposs, qvels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="downloads")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, rewards, ticks, qposs, qvels = run_rollout(args.duration, args.seed)
    video_path = output_dir / "cvae_cheetah_rollout.mp4"
    first_frame_path = output_dir / "cvae_cheetah_first_frame.png"
    log_path = output_dir / "cvae_cheetah_rollout_log.png"

    imageio.mimsave(video_path, frames, fps=25)
    imageio.imwrite(first_frame_path, frames[0])
    save_log_plot(log_path, ticks, rewards, qposs, qvels)

    print(f"Wrote {video_path}")
    print(f"Wrote {first_frame_path}")
    print(f"Wrote {log_path}")
    print(f"Total reward: {sum(rewards):.2f}")
    print(f"Average reward: {np.mean(rewards):.4f}")


if __name__ == "__main__":
    main()
