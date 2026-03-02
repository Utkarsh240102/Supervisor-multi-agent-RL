"""
Quick Start Guide - How to Save and Test Different Models

After your current training finishes, follow these steps:
"""

from experiment_manager import ExperimentManager, save_current_training

# ==============================================================================
# STEP 1: Save Your Current Training (After 500 episodes complete)
# ==============================================================================

print("="*80)
print("STEP 1: SAVE CURRENT TRAINING")
print("="*80)

# Define what you trained
baseline_config = {
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
    'reward_function': 'default: -(queue + 0.5*waiting + 10*switching)'
}

# Save it
exp1_id = save_current_training(
    name='baseline_500ep',
    description='Initial training with default hyperparameters (GPU trained)',
    config=baseline_config
)

print(f"\n✓ Your baseline model is saved as: {exp1_id}")
print("  You can test it anytime with:")
print(f"  python main.py --mode evaluate --model-path experiments/{exp1_id}/model.pth --eval-episodes 100")

# ==============================================================================
# STEP 2: Make an Improvement (Example)
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: TRY AN IMPROVEMENT")
print("="*80)
print("\nExample: Train with improved reward function")
print("\n1. Edit sumo_environment.py, _calculate_reward() method:")
print("   Change: reward = -(total_queue + 2.0 * avg_waiting_time + 5 * switching_penalty)")
print("\n2. Train again:")
print("   $env:PYTHONDONTWRITEBYTECODE=\"1\"; python main.py --mode train --episodes 500")
print("\n3. Save the new experiment:")

# After second training completes:
improved_config = baseline_config.copy()
improved_config['reward_function'] = 'modified: -(queue + 2.0*waiting + 5*switching)'

exp2_id = save_current_training(
    name='reward_tuned_v1',
    description='Increased waiting time penalty to 2.0, reduced switching penalty to 5',
    config=improved_config
)

print(f"\n✓ Improved model saved as: {exp2_id}")

# ==============================================================================
# STEP 3: Compare All Experiments
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: COMPARE EXPERIMENTS")
print("="*80)

manager = ExperimentManager()

# List all experiments
print("\nAll saved experiments:")
manager.list_experiments()

# Compare them visually
print("\nGenerating comparison plots...")
all_exp_ids = list(manager.experiments.keys())
if len(all_exp_ids) >= 2:
    manager.compare_experiments(all_exp_ids[:2])  # Compare first 2

# Find best one
print("\nFinding best experiment...")
best_by_reward = manager.get_best_experiment(metric='avg_reward')
best_by_waiting = manager.get_best_experiment(metric='avg_waiting_time')
best_by_queue = manager.get_best_experiment(metric='avg_queue')

# ==============================================================================
# STEP 4: Test Any Saved Model
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: TEST ANY SAVED MODEL")
print("="*80)

# Example: Test the baseline model
model_path = manager.load_experiment(exp1_id)
print(f"\nTo evaluate this model:")
print(f"python main.py --mode evaluate --model-path {model_path} --eval-episodes 100 --gui")

# Example: Test episode 400 checkpoint from any experiment
print(f"\nTo test episode 400 checkpoint:")
print(f"python main.py --mode evaluate --model-path experiments/{exp1_id}/checkpoints/ddqn_episode_400.pth --eval-episodes 50")

# ==============================================================================
# QUICK COMMANDS CHEATSHEET
# ==============================================================================

print("\n" + "="*80)
print("QUICK COMMANDS CHEATSHEET")
print("="*80)

cheatsheet = """
# Save current training after it completes:
python -c "from experiment_manager import save_current_training; save_current_training('my_exp', 'description', {'learning_rate': 0.001})"

# List all experiments:
python -c "from experiment_manager import ExperimentManager; ExperimentManager().list_experiments()"

# Compare all experiments (generates plots):
python -c "from experiment_manager import compare_all_experiments; compare_all_experiments()"

# Test a specific experiment:
python main.py --mode evaluate --model-path experiments/EXPERIMENT_ID/model.pth --eval-episodes 100

# Test with SUMO GUI:
python main.py --mode evaluate --model-path experiments/EXPERIMENT_ID/model.pth --eval-episodes 10 --gui

# Test a specific checkpoint (e.g., episode 400):
python main.py --mode evaluate --model-path checkpoints/ddqn_episode_400.pth --eval-episodes 50

# Train a new model:
$env:PYTHONDONTWRITEBYTECODE="1"; python main.py --mode train --episodes 500

# Train for longer:
$env:PYTHONDONTWRITEBYTECODE="1"; python main.py --mode train --episodes 1000
"""

print(cheatsheet)

# ==============================================================================
# RECOMMENDED WORKFLOW
# ==============================================================================

print("\n" + "="*80)
print("RECOMMENDED WORKFLOW FOR IMPROVEMENTS")
print("="*80)

workflow = """
1. Wait for current 500-episode training to finish (should be done soon!)

2. Save baseline:
   >>> from experiment_manager import save_current_training
   >>> save_current_training('baseline', 'GPU trained 500 episodes', {'episodes': 500})

3. Evaluate baseline:
   >>> python main.py --mode evaluate --eval-episodes 100

4. Check results in results/ folder

5. Make ONE change (see IMPROVEMENT_GUIDE.md for ideas)

6. Train again:
   >>> python main.py --mode train --episodes 500

7. Save new experiment with descriptive name:
   >>> save_current_training('experiment_2', 'what I changed', {...})

8. Evaluate and compare:
   >>> python main.py --mode evaluate --eval-episodes 100
   >>> from experiment_manager import ExperimentManager
   >>> ExperimentManager().compare_experiments(['baseline_...', 'experiment_2_...'])

9. Repeat steps 5-8 until satisfied!

10. Use best model for final demonstration
"""

print(workflow)

print("\n" + "="*80)
print("Your current training is almost done! When it finishes:")
print("1. Run this script: python save_and_test_guide.py")
print("2. Follow the steps above to save and test your model")
print("3. Read IMPROVEMENT_GUIDE.md for ideas to improve performance")
print("="*80)
