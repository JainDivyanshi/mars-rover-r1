import sys
import os

sys.path.append(os.path.abspath("src/environment"))
sys.path.append(os.path.abspath("src"))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
import numpy as np

from mars_env import MarsEnv

TERRAIN_PATH = "data/terrain/mars.tif"
N_ENVS = 4          # 4 parallel rovers — 4x faster gradient estimation
TOTAL_STEPS = 500_000  # more steps needed for stable learning


# ─── Custom callback: prints curriculum stage and success rate ────────────────
class CurriculumLogger(BaseCallback):
    """
    Logs success rate (goal reached / total episodes) every N steps.
    Helps you see if the rover is actually learning.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.successes = 0
        self.total_eps = 0

    def _on_step(self) -> bool:
        # SB3 stores episode info in self.locals['infos']
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                self.episode_rewards.append(ep_rew)
                self.total_eps += 1
                # A success = episode reward above threshold
                # (goal bonus is 20, so any ep with mean > 5 likely reached goal)
                if ep_rew > 5.0:
                    self.successes += 1

        # Log every 10,000 steps
        if self.n_calls % 10_000 == 0 and self.total_eps > 0:
            recent = self.episode_rewards[-50:] if len(
                self.episode_rewards) >= 50 else self.episode_rewards
            success_rate = self.successes / self.total_eps * 100
            print(
                f"\n[Step {self.n_calls:>7}] "
                f"Mean ep reward (last 50): {np.mean(recent):+.2f} | "
                f"Success rate: {success_rate:.1f}% ({self.successes}/{self.total_eps})"
            )
        return True


def make_env_fn(terrain_path, seed=0):
    """Factory function — returns a callable that creates one monitored env."""
    def _init():
        env = MarsEnv(terrain_path=terrain_path, render=False)
        env = Monitor(env)  # wraps env to track episode stats
        return env
    return _init


if __name__ == "__main__":
    print("=" * 55)
    print("  Mars Rover RL Training")
    print(f"  Terrain : {TERRAIN_PATH}")
    print(f"  Envs    : {N_ENVS} parallel")
    print(f"  Steps   : {TOTAL_STEPS:,}")
    print(f"  Device  : CPU")
    print("=" * 55)

    # ── Build vectorized environment ──────────────────────────────────────────
    # DummyVecEnv runs envs sequentially in same process (safe on Windows)
    # SubprocVecEnv is faster but can have issues on Windows with some setups
    # Start with DummyVecEnv — if it's stable, switch to SubprocVecEnv
    env_fns = [make_env_fn(TERRAIN_PATH, seed=i) for i in range(N_ENVS)]
    env = DummyVecEnv(env_fns)

    # ── Separate eval environment (1 env, no parallel) ───────────────────────
    eval_env = DummyVecEnv([make_env_fn(TERRAIN_PATH, seed=99)])

    # ── PPO hyperparameters ───────────────────────────────────────────────────
    # Key changes from original:
    # - n_steps=512 (was 2048): shorter rollouts = more frequent updates
    #   With 4 envs, total batch = 4 * 512 = 2048 samples per update (same!)
    #   but updates happen 4x more often → smoother learning
    # - ent_coef=0.01: entropy bonus prevents premature convergence
    #   (stops rover from getting stuck in "spin in place" local minima)
    # - vf_coef=0.5: value function coefficient
    # - clip_range=0.2: standard PPO clip, keeps updates conservative
    # - target_kl=0.05: early stopping if policy changes too fast per update
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,           # per env per update (was 2048)
        batch_size=256,        # mini-batch size (was 64)
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,       # GAE smoothing — reduces variance
        clip_range=0.2,
        ent_coef=0.01,         # entropy bonus — prevents spinning in place
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,        # stops update if KL divergence too high
        device="cpu",
        policy_kwargs={
            "net_arch": [256, 256, 128]  # deeper than default [64, 64]
        },
        tensorboard_log="./tensorboard_logs/"
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # 1. Save checkpoint every 25,000 steps
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000 // N_ENVS,  # divide by n_envs (SB3 counts per env)
        save_path="./models/checkpoints/",
        name_prefix="rover",
        verbose=1
    )

    # 2. Evaluate on separate env every 10,000 steps, save best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./tensorboard_logs/eval/",
        eval_freq=10_000 // N_ENVS,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )

    # 3. Custom curriculum + success rate logger
    curriculum_cb = CurriculumLogger(verbose=1)

    # ── Train! ────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    print("Open TensorBoard in another terminal:")
    print("  tensorboard --logdir tensorboard_logs\n")

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[checkpoint_cb, eval_cb, curriculum_cb],
            progress_bar=True,
            reset_num_timesteps=True,
            tb_log_name="ppo_mars_rover"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current model...")

    # ── Save final model ──────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    model.save("models/mars_rover_final")
    print("\nModel saved → models/mars_rover_final.zip")
    print("Best model → models/best/best_model.zip")
    print("\nTo watch your rover drive:")
    print("  python src/rl_agent/evaluate.py")