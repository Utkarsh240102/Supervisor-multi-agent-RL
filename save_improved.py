"""
Save the IMPROVED training run after completion
"""

from experiment_manager import save_current_training

# Configuration of improved version
config = {
    'episodes': 500,
    'learning_rate': 0.001,
    'gamma': 0.99,  # Changed from 0.95
    'hidden_dim': 256,  # Changed from 128
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'batch_size': 64,
    'buffer_capacity': 10000,
    'target_update_freq': 10,
    'state_dim': 14,  # Changed from 6
    'action_dim': 2,
    'reward_function': 'improved: -(queue + 2.0*waiting)',  # Changed from 0.5 to 2.0
    'state_features': 'Enhanced: queues(4) + speeds(4) + vehicles(4) + phase + time',
    'device': 'cuda (RTX 4060)'
}

# Save the experiment
exp_id = save_current_training(
    name='improved_v1_multi',
    description='Multiple improvements: reward(0.5→2.0), state(6→14), hidden(128→256), gamma(0.95→0.99)',
    config=config
)

print("\n" + "="*80)
print("✓ IMPROVED TRAINING SAVED!")
print("="*80)
print(f"\nExperiment ID: {exp_id}")
print(f"\nLocation: experiments/{exp_id}/")
print("\nImprovements Made:")
print("  1. ✓ Reward: Waiting penalty 0.5 → 2.0 (prioritize flow)")
print("  2. ✓ State: 6 → 14 features (added speeds + vehicle counts)")
print("  3. ✓ Network: 128 → 256 hidden units (more capacity)")
print("  4. ✓ Gamma: 0.95 → 0.99 (better long-term planning)")
print("\nHow to test this model:")
print(f"  python main.py --mode evaluate --model-path experiments/{exp_id}/model.pth --eval-episodes 10 --gui")
print("\nCompare with baseline:")
print(f"  from experiment_manager import ExperimentManager")
print(f"  manager = ExperimentManager()")
print(f"  manager.compare_experiments(['baseline_500ep_GPU_20260205_235635', '{exp_id}'])")
print("\n" + "="*80)
