import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication-grade aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

os.makedirs('figures', exist_ok=True)

# Full 7-Phase SCATS Benchmark Data
phases = [
    "1. Webster Fixed-Time\n(1 Intersection)",
    "2. Single Agent DDQN\n(1 Intersection)",
    "3. 4-Inter Independent\nDDQN (2x2 Grid)",
    "4. 4-Inter Cooperative\nDDQN (2x2 Grid)",
    "5. 8-Inter Independent\nDDQN (2x4 Grid)",
    "6. 8-Inter Supervisor\n(No Comm, 2x4 Grid)",
    "7. 8-Inter Supervisor\n(With Comm, 2x4 Grid)"
]

throughput = [621.00, 480.00, 334.00, 384.00, 431.33, 440.67, 462.67]
throughput_std = [0.00, 0.00, 0.00, 0.00, 34.99, 22.07, 27.35]

# Distinct colors by network tier
colors = [
    '#64748B',  # Baseline grey
    '#0EA5E9',  # Single RL sky blue
    '#818CF8',  # 4-inter indep purple
    '#4F46E5',  # 4-inter coop deep purple
    '#F43F5E',  # 8-inter indep rose
    '#E11D48',  # 8-inter sup crimson
    '#9F1239'   # 8-inter sup+comm burgundy
]

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
fig.patch.set_facecolor('white')

x = np.arange(len(phases))
bars = ax.bar(x, throughput, yerr=throughput_std, capsize=6, color=colors, width=0.58, edgecolor='none', alpha=0.92)

ax.set_title("Victoria SCATS Real-World Benchmark: Throughput Across All 7 Architectural Phases", 
             fontsize=14, fontweight='bold', pad=20, color='#0F172A')
ax.set_ylabel("Throughput (Completed Trips per Hour) ↑", fontsize=12, fontweight='bold', color='#334155')
ax.set_xticks(x)
ax.set_xticklabels(phases, rotation=22, ha='right', fontsize=10, fontweight='bold', color='#334155')
ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E1')
ax.set_ylim(0, 720)

# Value annotations above bars
for idx, bar in enumerate(bars):
    h = bar.get_height()
    std = throughput_std[idx]
    label_y = h + std + 14
    ax.text(bar.get_x() + bar.get_width()/2.0, label_y, f"{h:.1f}", 
            ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#0F172A')

# Annotate Cooperative Gain (+15%)
ax.annotate("+15.0% Boost\nvia Cooperation", xy=(3, 384), xytext=(2.3, 510),
            arrowprops=dict(arrowstyle="->", color="#4F46E5", lw=2),
            fontsize=10, fontweight='bold', color='#4F46E5', ha='center',
            bbox=dict(boxstyle="round,pad=0.4", fc="#EEF2FF", ec="#4F46E5", lw=1.2))

# Annotate Supervisor Communication Gain (+31.3 trips vs Indep)
ax.annotate("Highest 8-Inter Flow\n(+31.3 Trips vs Indep)", xy=(6, 462.7), xytext=(5.1, 610),
            arrowprops=dict(arrowstyle="->", color="#9F1239", lw=2),
            fontsize=10, fontweight='bold', color='#9F1239', ha='center',
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF1F2", ec="#9F1239", lw=1.2))

plt.tight_layout()
output_file = os.path.join("figures", "throughput_comparison_all_phases.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully created throughput graph: {output_file}")
