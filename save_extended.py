"""
Save the EXTENDED baseline training (1000 episodes)
"""

from experiment_manager import save_current_training

# Configuration - same as baseline, just more episodes
config = {
    'episodes': 1000,
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
    'training_time': '~150 minutes',
    'device': 'cuda (RTX 4060)',
    'notes': 'Extended training of proven baseline configuration'
}

# Save the experiment
exp_id = save_current_training(
    name='baseline_1000ep_extended',
    description='Extended baseline training - 1000 episodes with proven hyperparameters (not the failed improvements)',
    config=config
)

print("\n" + "="*80)
print("✓ EXTENDED BASELINE SAVED!")
print("="*80)
print(f"\nExperiment ID: {exp_id}")
print(f"\nLocation: experiments/{exp_id}/")
print("\nYour Saved Models:")
print("  1. Baseline 500ep: experiments/baseline_500ep_GPU_20260205_235635/")
print(f"  2. Extended 1000ep: experiments/{exp_id}/")
print("\nHow to compare them:")
print("  from experiment_manager import ExperimentManager")
print("  manager = ExperimentManager()")
print("  manager.compare_experiments([")
print("      'baseline_500ep_GPU_20260205_235635',")
print(f"      '{exp_id}'")
print("  ])")
print("\n" + "="*80)
