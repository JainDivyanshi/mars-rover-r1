"""
evaluate.py
───────────
Load the best trained model and watch the rover drive on Mars.
Run from project root:
    python src/rl_agent/evaluate.py
"""
import sys
import os

sys.path.append(os.path.abspath("src/environment"))
sys.path.append(os.path.abspath("src"))

from stable_baselines3 import PPO
from mars_env import MarsEnv
import numpy as np


TERRAIN_PATH = "data/terrain/mars.tif"
N_EPISODES   = 10    # how many episodes to watch
USE_BEST     = True  # True = use best model, False = use final model


def run_eval(model_path: str, n_episodes: int = 10, render: bool = True):
    env = MarsEnv(terrain_path=TERRAIN_PATH, render=render)
    model = PPO.load(model_path)

    print(f"\nLoaded model: {model_path}")
    print(f"Running {n_episodes} episodes with render={'GUI' if render else 'DIRECT'}\n")

    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        reached_goal = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                # Check if goal was reached (reward spike > 10 means goal bonus)
                if total_reward > 5.0:
                    reached_goal = True
                break

        status = "✓ GOAL" if reached_goal else "✗ timeout"
        print(
            f"Episode {ep+1:>2} | {status} | "
            f"Steps: {steps:>4} | "
            f"Reward: {total_reward:>8.2f} | "
            f"Goal range: {env._get_goal_range()}m"
        )
        results.append({
            "steps": steps,
            "reward": total_reward,
            "success": reached_goal
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    success_rate = sum(r["success"] for r in results) / len(results) * 100
    mean_reward  = np.mean([r["reward"] for r in results])
    mean_steps   = np.mean([r["steps"] for r in results])

    print("\n" + "=" * 50)
    print(f"  Success rate : {success_rate:.0f}%  ({sum(r['success'] for r in results)}/{n_episodes})")
    print(f"  Mean reward  : {mean_reward:.2f}")
    print(f"  Mean steps   : {mean_steps:.0f}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    # Try best model first, fall back to final
    if USE_BEST and os.path.exists("models/best/best_model.zip"):
        model_path = "models/best/best_model"
    elif os.path.exists("models/mars_rover_final.zip"):
        model_path = "models/mars_rover_final"
    else:
        print("No trained model found.")
        print("Run: python src/rl_agent/train.py first.")
        sys.exit(1)

    run_eval(model_path, n_episodes=N_EPISODES, render=True)