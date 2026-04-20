import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results_security"
ANALYSIS_DIR = "analysis_security"

# Ensure output directory exists
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Standard styling for academic charts
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'baseline': '#1f77b4',   # Blue
    'attack': '#d62728',     # Red
    'defense': '#2ca02c',    # Green
    'unreliable': '#ff7f0e', # Orange
    'secure': '#9467bd'      # Purple
}

def plot_average_reward(summary_df):
    """Chart 1: Average Reward Comparison with Error Bars"""
    print("Generating Chart 1: Average Reward Comparison...")
    plt.figure(figsize=(10, 6))
    
    scenarios = summary_df['scenario'].tolist()
    rewards = summary_df['avg_network_reward'].tolist()
    stds = summary_df['std_network_reward'].tolist()
    
    colors_list = [COLORS.get(s, '#333333') for s in scenarios]
    
    # Create bar chart with standard deviation error bars
    bars = plt.bar(scenarios, rewards, yerr=stds, capsize=8, color=colors_list, alpha=0.85, edgecolor='black')
    
    # Add a horizontal dashed line representing the 'baseline' performance
    baseline_reward = float(summary_df.loc[summary_df['scenario'] == 'baseline', 'avg_network_reward'].iloc[0])
    plt.axhline(y=baseline_reward, color='black', linestyle='--', alpha=0.6, label='Baseline Performance')
    
    # Labels and titles
    plt.title('System Resilience: Average Network Reward under Attack and Defense', fontsize=14, fontweight='bold')
    plt.ylabel('Average Reward (Higher is Better)', fontsize=12)
    plt.xlabel('Experiment Scenario', fontsize=12)
    plt.xticks(ticks=range(len(scenarios)), labels=[s.capitalize() for s in scenarios], fontsize=11)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    output_path = os.path.join(ANALYSIS_DIR, '01_average_reward_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_average_wait_time(summary_df):
    """Chart 2: Average Wait Time Comparison"""
    print("Generating Chart 2: Average Wait Time Comparison...")
    plt.figure(figsize=(10, 6))
    
    scenarios = summary_df['scenario'].tolist()
    waits = summary_df['avg_wait_time'].tolist()
    stds = summary_df['std_wait_time'].tolist()
    
    colors_list = [COLORS.get(s, '#333333') for s in scenarios]
    
    # Create bar chart with standard deviation error bars
    bars = plt.bar(scenarios, waits, yerr=stds, capsize=8, color=colors_list, alpha=0.85, edgecolor='black')
    
    # Add a horizontal dashed line representing the 'baseline' performance
    baseline_wait = float(summary_df.loc[summary_df['scenario'] == 'baseline', 'avg_wait_time'].iloc[0])
    plt.axhline(y=baseline_wait, color='black', linestyle='--', alpha=0.6, label='Baseline Performance')
    
    # Labels and titles
    plt.title('Congestion Impact: Average Waiting Time per Vehicle', fontsize=14, fontweight='bold')
    plt.ylabel('Waiting Time (Lower is Better)', fontsize=12)
    plt.xlabel('Experiment Scenario', fontsize=12)
    plt.xticks(ticks=range(len(scenarios)), labels=[s.capitalize() for s in scenarios], fontsize=11)
    
    # Put legend in optimal location
    plt.legend(loc='upper left')
    
    # Save the plot
    plt.tight_layout()
    output_path = os.path.join(ANALYSIS_DIR, '02_average_wait_time_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_per_intersection_breakdown():
    """Chart 3: Per-intersection reward breakdown (Attack vs Defense)"""
    print("Generating Chart 3: Per-Intersection Breakdown...")
    
    attack_path = os.path.join(RESULTS_DIR, 'attack_results.csv')
    defense_path = os.path.join(RESULTS_DIR, 'defense_results.csv')
    
    if not (os.path.exists(attack_path) and os.path.exists(defense_path)):
        print(f"Error: Missing {attack_path} or {defense_path}")
        return
        
    attack_df = pd.read_csv(attack_path)
    defense_df = pd.read_csv(defense_path)
    
    tls_cols = [f'tls_{i}' for i in range(1, 9)]
    attack_means = attack_df[tls_cols].mean()
    defense_means = defense_df[tls_cols].mean()
    
    # Setup the figure
    plt.figure(figsize=(12, 6))
    x = np.arange(len(tls_cols))
    width = 0.35
    
    # Plot grouped bars
    plt.bar(x - width/2, attack_means, width, label='Attack (No Defense)', color=COLORS['attack'], alpha=0.85, edgecolor='black')
    plt.bar(x + width/2, defense_means, width, label='Defense (LSTM Active)', color=COLORS['defense'], alpha=0.85, edgecolor='black')
    
    # Labels and formatting
    plt.title('Vulnerability Map: Reward per Intersection (Attack vs Defense)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Reward (Higher is Better)', fontsize=12)
    plt.xlabel('Traffic Light Intersection', fontsize=12)
    plt.xticks(x, [f'TLS {i}' for i in range(1, 9)], fontsize=11)
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(ANALYSIS_DIR, '03_per_intersection_breakdown.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    print(f"Loading data from {RESULTS_DIR}...")
    summary_path = os.path.join(RESULTS_DIR, "scenario_comparison.csv")
    
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Run main_security.py first.")
        return
        
    summary_df = pd.read_csv(summary_path)
    
    # Generate charts
    plot_average_reward(summary_df)
    plot_average_wait_time(summary_df)
    plot_per_intersection_breakdown()
    
    print(f"\nAll charts saved successfully to {ANALYSIS_DIR}/")

if __name__ == "__main__":
    main()
