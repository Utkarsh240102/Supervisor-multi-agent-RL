"""
Save the current baseline training run
"""

from experiment_manager import save_current_training

# Configuration of what was trained
config = {
    'episodes': 500,
    'learning_rate': 0.001,
    'gamma': 0.95,
    'hidden_dim': 128,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'batch_size': 64,
    'buffer_capacity': 10000,
    'target_update_freq': 10,
    'state_dim': 6,
    'action_dim': 2,
    'reward_function': 'default: -(queue + 0.5*waiting + 10*switching)',
    'training_time': '~75 minutes',
    'device': 'cuda (RTX 4060)'
}

# Save the experiment
exp_id = save_current_training(
    name='baseline_500ep_GPU',
    description='First successful GPU training - 500 episodes, 61% improvement from -17308 to -6670 avg reward',
    config=config
)

print("\n" + "="*80)
print("✓ YOUR TRAINING IS NOW PERMANENTLY SAVED!")
print("="*80)
print(f"\nExperiment ID: {exp_id}")
print(f"\nLocation: experiments/{exp_id}/")
print("\nWhat was saved:")
print("  ✓ Final model: model.pth")
print("  ✓ Episode 400 checkpoint: checkpoint_400.pth")
print("  ✓ All checkpoints: checkpoints/ folder (100, 200, 300, 400, 500)")
print("  ✓ Training history: training_history.csv")
print("\nHow to test this specific model later:")
print(f"  python main.py --mode evaluate --model-path experiments/{exp_id}/model.pth --eval-episodes 10 --gui")
print("\nHow to test episode 400 checkpoint:")
print(f"  python main.py --mode evaluate --model-path experiments/{exp_id}/checkpoint_400.pth --eval-episodes 10 --gui")
print("\n" + "="*80)
