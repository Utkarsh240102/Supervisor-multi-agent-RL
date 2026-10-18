import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication-grade aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

# Ensure output folder exists
os.makedirs('figures', exist_ok=True)

# Data from Option B (Verified 60-Minute SCATS Benchmark)
phases = [
    "1. Webster Fixed-Time\n(1-Inter Baseline)",
    "2. Single Agent DDQN\n(1-Inter)",
    "3. 4-Inter Independent\nDDQN",
    "4. 4-Inter Cooperative\nDDQN",
    "5. 8-Inter Independent\nDDQN",
    "6. 8-Inter Supervisor\n(No Comm)",
    "7. 8-Inter Supervisor\n(With Comm)"
]

throughput = [621.00, 480.00, 334.00, 384.00, 431.33, 440.67, 462.67]
throughput_std = [0.00, 0.00, 0.00, 0.00, 34.99, 22.07, 27.35]

wait_time = [1.73, 0.20, 0.04, 0.04, 0.04, 0.03, 0.03]

# Create a stunning 1x2 subplots figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
fig.patch.set_facecolor('white')

# Colors: distinguish baselines, independent, cooperative, and hierarchical supervisors
colors_t = ['#94A3B8', '#38BDF8', '#818CF8', '#4F46E5', '#F43F5E', '#E11D48', '#BE123C']
colors_w = ['#EF4444', '#F97316', '#10B981', '#059669', '#06B6D4', '#0284C7', '#2563EB']

# -------------------------------------------------------------------------
# Panel 1: Throughput (Completed Trips)
# -------------------------------------------------------------------------
x = np.arange(len(phases))
bars1 = ax1.bar(x, throughput, yerr=throughput_std, capsize=4, color=colors_t, width=0.65, edgecolor='none', alpha=0.92)

ax1.set_title("A. Network Throughput (Completed Trips per Hour) ↑", fontsize=13, fontweight='bold', pad=15, color='#1E293B')
ax1.set_ylabel("Completed Trips", fontsize=11, fontweight='bold', color='#334155')
ax1.set_xticks(x)
ax1.set_xticklabels(phases, rotation=30, ha='right', fontsize=9.5, color='#334155')
ax1.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E1')
ax1.set_ylim(0, 720)

# Add value annotations
for idx, bar in enumerate(bars1):
    height = bar.get_height()
    std = throughput_std[idx]
    label = f"{height:.1f}"
    y_pos = height + std + 12
    ax1.text(bar.get_x() + bar.get_width()/2.0, y_pos, label, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1E293B')

# Add annotation arrows for key highlights
ax1.annotate("+15.0% Boost\nvia Cooperation", xy=(3, 384), xytext=(2.3, 490),
             arrowprops=dict(arrowstyle="->", color="#4F46E5", lw=1.5),
             fontsize=9, fontweight='bold', color='#4F46E5', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="#EEF2FF", ec="#4F46E5", lw=1))

ax1.annotate("Highest 8-Inter Flow\n(+31.3 Trips vs Indep)", xy=(6, 462.7), xytext=(5.1, 570),
             arrowprops=dict(arrowstyle="->", color="#BE123C", lw=1.5),
             fontsize=9, fontweight='bold', color='#BE123C', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFF1F2", ec="#BE123C", lw=1))

# -------------------------------------------------------------------------
# Panel 2: 60-Minute Average Waiting Time
# -------------------------------------------------------------------------
bars2 = ax2.bar(x, wait_time, color=colors_w, width=0.65, edgecolor='none', alpha=0.92)

ax2.set_title("B. 60-Minute Continuous Average Waiting Time (s) ↓", fontsize=13, fontweight='bold', pad=15, color='#1E293B')
ax2.set_ylabel("Avg Wait Time per Vehicle (Seconds)", fontsize=11, fontweight='bold', color='#334155')
ax2.set_xticks(x)
ax2.set_xticklabels(phases, rotation=30, ha='right', fontsize=9.5, color='#334155')
ax2.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E1')
ax2.set_ylim(0, 2.05)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, height + 0.04, f"{height:.2f}s", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1E293B')

ax2.annotate("88.4% Delay Reduction\nvs Fixed-Time", xy=(1, 0.20), xytext=(2.2, 1.2),
             arrowprops=dict(arrowstyle="->", color="#EA580C", lw=1.5),
             fontsize=9, fontweight='bold', color='#EA580C', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFF7ED", ec="#EA580C", lw=1))

plt.tight_layout()
output_path = os.path.join("figures", "scats_sample_throughput_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated sample figure: {output_path}")
