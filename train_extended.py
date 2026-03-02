"""
Train Extended Baseline - 1000 Episodes
Same proven configuration, just train longer for better convergence
"""

print("="*80)
print("EXTENDED BASELINE TRAINING - 1000 EPISODES")
print("="*80)
print("\nUsing the PROVEN baseline configuration:")
print("  - STATE_DIM: 6 (queues + phase + time)")
print("  - HIDDEN_DIM: 128")
print("  - GAMMA: 0.95")
print("  - Reward: -(queue + 0.5*waiting)")
print("\nWhy this works:")
print("  ✓ 500 episodes showed 61% improvement (-17,308 → -6,670)")
print("  ✓ Agent was still improving at episode 500")
print("  ✓ More episodes = better convergence & performance")
print("\nEstimated training time: ~2.5 hours")
print("\nStarting training in 5 seconds...")
print("="*80)

import time
time.sleep(5)

import os
os.system('python main.py --mode train --episodes 1000')
