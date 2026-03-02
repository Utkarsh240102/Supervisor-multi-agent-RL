"""
Comparison Visuals: 4×1 Cooperative vs 4×2 Grouped Cooperative
- Training curves from real 4-intersection data
- Architecture layout diagrams
- Per-intersection performance
- Network scaling analysis
- Saves all figures to results_8intersection/comparison/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, ConnectionPatch
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = 'results_8intersection/comparison'
DATA_4INT   = 'results_cooperative/training_history.csv'
DATA_8INT   = 'results_8intersection/pretrain_eval_results.csv'   # optional

COLORS = {
    'group_a':    '#45B7D1',
    'group_b':    '#9B59B6',
    'network_4':  '#2ECC71',
    'network_8':  '#E67E22',
    'highlight':  '#E74C3C',
    'bg':         '#F8F9FA',
    'grid':       '#DEE2E6',
}


# ────────────────────────────────────────────────────────────────────
def load_4int_data():
    df = pd.read_csv(DATA_4INT)
    df.columns = ['network_reward', 'tls1', 'tls2', 'tls3', 'tls4']
    df['episode'] = range(1, len(df) + 1)
    # Smoothed with 20-episode rolling average
    df['smooth_network'] = df['network_reward'].rolling(20, min_periods=1).mean()
    df['smooth_per_int'] = df['tls1'].rolling(20, min_periods=1).mean()
    return df


def load_8int_eval(df4):
    """Load 8-int eval if available, else project from 4-int final performance."""
    if os.path.exists(DATA_8INT):
        df8 = pd.read_csv(DATA_8INT)
        return df8, False
    # Project: each group replicates 4-int final performance
    last50 = df4.tail(50)
    per_int = last50['tls1'].mean()
    proj = {
        'tls_1': per_int, 'tls_2': per_int, 'tls_3': per_int, 'tls_4': per_int,
        'tls_5': per_int, 'tls_6': per_int, 'tls_7': per_int, 'tls_8': per_int,
    }
    return proj, True   # (dict, is_projected)


# ════════════════════════════════════════════════════════════════════
# Figure 1: Architecture Comparison Diagram
# ════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor(COLORS['bg'])

    def draw_intersection(ax, x, y, label, color, size=0.35, fontsize=11):
        box = FancyBboxPatch((x - size/2, y - size/2), size, size,
                             boxstyle="round,pad=0.04",
                             facecolor=color, edgecolor='white',
                             linewidth=2.5, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white', zorder=4)

    def draw_road(ax, x1, y1, x2, y2, color='#ADB5BD', lw=3):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
                zorder=1, solid_capstyle='round')

    def draw_coop_link(ax, x1, y1, x2, y2, color, lw=2.5, style='--'):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
                linestyle=style, zorder=2, alpha=0.9)

    def axis_setup(ax, title):
        ax.set_facecolor(COLORS['bg'])
        ax.set_xlim(-0.8, 3.8)
        ax.set_ylim(-0.8, 3.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # ── Left: 4×1 (2×2 grid, 4 intersections) ──────────────────────
    axis_setup(ax1, '4-Intersection Cooperative (4×1)\n'
               'Single Region · Full Cooperation')
    pos4 = {
        'tls_1': (0.6, 2.4), 'tls_2': (2.4, 2.4),
        'tls_3': (0.6, 0.6), 'tls_4': (2.4, 0.6),
    }
    # Road grid
    for (x1, y1), (x2, y2) in [
        (pos4['tls_1'], pos4['tls_2']), (pos4['tls_3'], pos4['tls_4']),
        (pos4['tls_1'], pos4['tls_3']), (pos4['tls_2'], pos4['tls_4']),
    ]:
        draw_road(ax1, x1, y1, x2, y2)
    # Border stubs
    for tls, (bx, by) in [('tls_1', (-0.3, 2.4)), ('tls_1', (0.6, 3.2)),
                           ('tls_2', (3.3, 2.4)), ('tls_2', (2.4, 3.2)),
                           ('tls_3', (-0.3, 0.6)), ('tls_3', (0.6, -0.2)),
                           ('tls_4', (3.3, 0.6)), ('tls_4', (2.4, -0.2))]:
        x, y = pos4[tls]
        draw_road(ax1, x, y, bx, by, color='#CED4DA', lw=2)
    # Cooperation dashed links
    for (a, b) in [('tls_1', 'tls_2'), ('tls_1', 'tls_3'),
                   ('tls_2', 'tls_4'), ('tls_3', 'tls_4')]:
        x1, y1 = pos4[a]; x2, y2 = pos4[b]
        draw_coop_link(ax1, x1, y1, x2, y2, COLORS['group_a'], lw=2, style=':')
    # Nodes
    for tls, (x, y) in pos4.items():
        draw_intersection(ax1, x, y, tls.replace('tls_', 'I'), COLORS['group_a'])

    # Region label
    rect = FancyBboxPatch((0, 0), 3, 3,
                          boxstyle="round,pad=0.15",
                          facecolor=COLORS['group_a'], alpha=0.08,
                          edgecolor=COLORS['group_a'], linewidth=2, linestyle='--')
    ax1.add_patch(rect)
    ax1.text(1.5, -0.55, 'Cooperative Region  (4 agents, shared reward)',
             ha='center', fontsize=11, color=COLORS['group_a'], fontweight='bold')
    ax1.text(1.5, 3.15, 'All 4 agents share reward & neighbor state',
             ha='center', fontsize=9.5, color='#666666', style='italic')

    # ── Right: 4×2 (2×4 grid, 8 intersections) ─────────────────────
    axis_setup(ax2, '8-Intersection Grouped Cooperative (4×2)\n'
               'Two Regions · Intra-Group Cooperation Only')
    ax2.set_xlim(-0.8, 6.4)
    ax2.set_ylim(-0.8, 3.3)

    pos8 = {
        # Group A (left 2×2)
        'tls_1': (0.6, 2.4), 'tls_2': (2.4, 2.4),
        'tls_3': (0.6, 0.6), 'tls_4': (2.4, 0.6),
        # Group B (right 2×2)
        'tls_5': (3.6, 2.4), 'tls_6': (5.4, 2.4),
        'tls_7': (3.6, 0.6), 'tls_8': (5.4, 0.6),
    }
    group_a_nodes = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
    group_b_nodes = ['tls_5', 'tls_6', 'tls_7', 'tls_8']

    # Group A road grid
    for a, b in [('tls_1', 'tls_2'), ('tls_3', 'tls_4'),
                 ('tls_1', 'tls_3'), ('tls_2', 'tls_4')]:
        x1, y1 = pos8[a]; x2, y2 = pos8[b]
        draw_road(ax2, x1, y1, x2, y2)
    # Group B road grid
    for a, b in [('tls_5', 'tls_6'), ('tls_7', 'tls_8'),
                 ('tls_5', 'tls_7'), ('tls_6', 'tls_8')]:
        x1, y1 = pos8[a]; x2, y2 = pos8[b]
        draw_road(ax2, x1, y1, x2, y2)
    # Cross-group roads (no cooperation — solid grey)
    for a, b in [('tls_2', 'tls_5'), ('tls_4', 'tls_7')]:
        x1, y1 = pos8[a]; x2, y2 = pos8[b]
        draw_road(ax2, x1, y1, x2, y2, color='#ADB5BD', lw=2)
    # Border stubs
    for tls, (bx, by) in [
        ('tls_1', (-0.3, 2.4)), ('tls_1', (0.6, 3.2)),
        ('tls_3', (-0.3, 0.6)), ('tls_3', (0.6, -0.2)),
        ('tls_6', (6.2, 2.4)),  ('tls_6', (5.4, 3.2)),
        ('tls_8', (6.2, 0.6)),  ('tls_8', (5.4, -0.2)),
        ('tls_2', (2.4, 3.2)),  ('tls_4', (2.4, -0.2)),
        ('tls_5', (3.6, 3.2)),  ('tls_7', (3.6, -0.2)),
    ]:
        x, y = pos8[tls]
        draw_road(ax2, x, y, bx, by, color='#CED4DA', lw=2)

    # Intra-group cooperation links
    for a, b in [('tls_1', 'tls_2'), ('tls_1', 'tls_3'),
                 ('tls_2', 'tls_4'), ('tls_3', 'tls_4')]:
        x1, y1 = pos8[a]; x2, y2 = pos8[b]
        draw_coop_link(ax2, x1, y1, x2, y2, COLORS['group_a'], lw=2, style=':')
    for a, b in [('tls_5', 'tls_6'), ('tls_5', 'tls_7'),
                 ('tls_6', 'tls_8'), ('tls_7', 'tls_8')]:
        x1, y1 = pos8[a]; x2, y2 = pos8[b]
        draw_coop_link(ax2, x1, y1, x2, y2, COLORS['group_b'], lw=2, style=':')
    # Cross-group barrier
    ax2.plot([3.0, 3.0], [-0.6, 3.1], color=COLORS['highlight'],
             linewidth=2, linestyle=(0, (4, 2)), alpha=0.5, zorder=5)
    ax2.text(3.0, 3.18, 'No cooperative\ninfo crossing',
             ha='center', va='bottom', fontsize=7.5,
             color=COLORS['highlight'], style='italic')

    # Group A region
    ra = FancyBboxPatch((0, 0), 3, 3, boxstyle="round,pad=0.15",
                        facecolor=COLORS['group_a'], alpha=0.08,
                        edgecolor=COLORS['group_a'], linewidth=2, linestyle='--')
    ax2.add_patch(ra)
    ax2.text(1.5, -0.55, 'Group A (I1–I4)',
             ha='center', fontsize=10, color=COLORS['group_a'], fontweight='bold')
    # Group B region
    rb = FancyBboxPatch((3.1, 0), 3, 3, boxstyle="round,pad=0.15",
                        facecolor=COLORS['group_b'], alpha=0.08,
                        edgecolor=COLORS['group_b'], linewidth=2, linestyle='--')
    ax2.add_patch(rb)
    ax2.text(4.6, -0.55, 'Group B (I5–I8)',
             ha='center', fontsize=10, color=COLORS['group_b'], fontweight='bold')

    # Nodes
    for tls, (x, y) in pos8.items():
        c = COLORS['group_a'] if tls in group_a_nodes else COLORS['group_b']
        label = 'I' + tls.split('_')[1]
        draw_intersection(ax2, x, y, label, c)

    ax2.text(3.0, 3.18, '', ha='center')  # placeholder

    # Legend
    legend_els = [
        mpatches.Patch(facecolor=COLORS['group_a'], label='Group A agents'),
        mpatches.Patch(facecolor=COLORS['group_b'], label='Group B agents'),
        plt.Line2D([0], [0], color='#ADB5BD', linestyle=':', linewidth=2,
                   label='Cooperation link'),
        plt.Line2D([0], [0], color=COLORS['highlight'], linestyle='--',
                   linewidth=1.5, label='Inter-group barrier'),
    ]
    ax2.legend(handles=legend_els, loc='upper right', fontsize=8.5,
               framealpha=0.9, edgecolor='#CED4DA')

    plt.suptitle('Traffic Light Control — Architecture Comparison',
                 fontsize=17, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = f'{OUTPUT_DIR}/architecture_comparison.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
    print(f"✅ Saved: {out}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# Figure 2: 4-Intersection Training Curve + Final Performance
# ════════════════════════════════════════════════════════════════════
def fig_training_curve(df4):
    fig = plt.figure(figsize=(16, 8))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    ax_main  = fig.add_subplot(gs[0, :2])
    ax_tls   = fig.add_subplot(gs[1, :2])
    ax_stats = fig.add_subplot(gs[:, 2])
    fig.patch.set_facecolor(COLORS['bg'])
    for ax in [ax_main, ax_tls, ax_stats]:
        ax.set_facecolor(COLORS['bg'])
        ax.grid(True, color=COLORS['grid'], linewidth=0.8)

    # ── Network reward ──
    ax_main.plot(df4['episode'], df4['network_reward'],
                 color=COLORS['network_4'], alpha=0.25, linewidth=1)
    ax_main.plot(df4['episode'], df4['smooth_network'],
                 color=COLORS['network_4'], linewidth=2.5, label='Network (smoothed)')
    ax_main.axhline(df4['network_reward'].tail(50).mean(), color='#E74C3C',
                    linestyle='--', linewidth=1.5, label=f'Final avg: {df4["network_reward"].tail(50).mean():.0f}')
    ax_main.set_title('4-Intersection Network Total Reward (700 Episodes)', fontweight='bold')
    ax_main.set_ylabel('Network Total Reward')
    ax_main.legend(fontsize=9)

    # ── Per-intersection reward ──
    colors_tls = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60']
    for i, col in enumerate(['tls1', 'tls2', 'tls3', 'tls4']):
        ax_tls.plot(df4['episode'], df4[col].rolling(20, min_periods=1).mean(),
                    color=colors_tls[i], linewidth=1.8,
                    label=f'TLS {i+1} (smoothed)')
    final_per = df4['tls1'].tail(50).mean()
    ax_tls.axhline(final_per, color='#888888', linestyle='--', linewidth=1.4,
                   label=f'Final avg/intersection: {final_per:.0f}')
    ax_tls.set_title('Per-Intersection Reward (all TLS equal — cooperative reward)', fontweight='bold')
    ax_tls.set_ylabel('Reward per Intersection')
    ax_tls.set_xlabel('Episode')
    ax_tls.legend(fontsize=8.5)

    # ── Stats panel ──
    ax_stats.axis('off')
    last50_net = df4['network_reward'].tail(50)
    last50_per = df4['tls1'].tail(50)
    stats = [
        ('4-Intersection Cooperative', '', '#1a1a2e'),
        ('700 Episodes, 2×2 Grid', '', '#444444'),
        ('', '', ''),
        ('Final Performance', '', COLORS['network_4']),
        (f"Network Avg (last 50): {last50_net.mean():.1f}", '', '#333333'),
        (f"Per-Int Avg (last 50): {last50_per.mean():.1f}", '', '#333333'),
        (f"Per-Int Std (last 50): ±{last50_per.std():.1f}", '', '#555555'),
        (f"Best episode:          {df4['network_reward'].min():.1f}", '', '#555555'),
        ('', '', ''),
        ('Architecture', '', COLORS['group_a']),
        ('Intersections:  4', '', '#333333'),
        ('Agents:          4 DDQN', '', '#333333'),
        ('State dim:      6 features', '', '#333333'),
        ('Reward:          Shared (group)', '', '#333333'),
        ('Neighbors:      2 per agent', '', '#333333'),
        ('', '', ''),
        ('Training Details', '', '#E67E22'),
        ('Episodes:         700', '', '#333333'),
        ('Epsilon final:   ~0.01', '', '#333333'),
        ('Checkpoint:     Final weights', '', '#333333'),
    ]
    y = 0.97
    for text, _, color in stats:
        if not text:
            y -= 0.030
            continue
        w = 'bold' if color not in ('#333333', '#555555', '#444444') else 'normal'
        ax_stats.text(0.05, y, text, transform=ax_stats.transAxes,
                      fontsize=9.5, color=color, fontweight=w, va='top',
                      fontfamily='monospace' if text[0].isspace() or ':' in text else 'sans-serif')
        y -= 0.045

    plt.suptitle('4-Intersection Cooperative DDQN — Training Analysis',
                 fontsize=14, fontweight='bold')
    out = f'{OUTPUT_DIR}/4int_training_analysis.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
    print(f"✅ Saved: {out}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# Figure 3: 4-Int vs 8-Int Performance Comparison
# ════════════════════════════════════════════════════════════════════
def fig_performance_comparison(df4, eval8, is_projected):
    last50 = df4['tls1'].tail(50)
    avg4 = last50.mean()
    std4 = last50.std()
    net4 = df4['network_reward'].tail(50).mean()

    if is_projected:
        # Project: 8-int mirrors 4-int per-intersection
        per_int_8 = {k: avg4 for k in eval8.keys()}
        std8 = std4
        net8 = avg4 * 8
        note = '(Projected — same weights as 4-int)'
    else:
        per_int_8 = {c: eval8[c].mean() for c in
                     ['tls_1','tls_2','tls_3','tls_4','tls_5','tls_6','tls_7','tls_8']}
        std8 = np.std(list(per_int_8.values()))
        net8 = eval8['network_reward'].mean()
        note = '(Measured — Pre-Fine-Tuning)'

    ga_avg = np.mean([per_int_8[f'tls_{i}'] for i in range(1, 5)])
    gb_avg = np.mean([per_int_8[f'tls_{i}'] for i in range(5, 9)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor(COLORS['bg'])
    for ax in axes:
        ax.set_facecolor(COLORS['bg'])
        ax.grid(axis='y', color=COLORS['grid'], linewidth=0.8)

    # ── Plot 1: Per-intersection bar ──
    ax = axes[0]
    labels4 = ['I1', 'I2', 'I3', 'I4']
    labels8 = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']
    vals4 = [avg4] * 4
    vals8 = [per_int_8[f'tls_{i}'] for i in range(1, 9)]
    x = np.arange(8)
    w = 0.38
    bars4 = ax.bar(x[:4] - w/2, vals4, w,
                   color=COLORS['network_4'], alpha=0.85,
                   edgecolor='black', linewidth=1.1, label='4-Int System')
    bars8_a = ax.bar(x[:4] + w/2, vals8[:4], w,
                     color=COLORS['group_a'], alpha=0.85,
                     edgecolor='black', linewidth=1.1, label='8-Int Group A')
    bars8_b = ax.bar(x[4:] + w/2 - 4, vals8[4:], w,
                     color=COLORS['group_b'], alpha=0.85,
                     edgecolor='black', linewidth=1.1, label='8-Int Group B')
    ax.set_xticks(x - 0.2)
    ax.set_xticklabels(labels8)
    ax.set_title('Per-Intersection Reward\n4-Int vs 8-Int (first 4 shared positions)', fontweight='bold', fontsize=10)
    ax.set_ylabel('Avg Reward per Intersection')
    ax.legend(fontsize=9, framealpha=0.9)
    for bar, v in zip(list(bars4) + list(bars8_a) + list(bars8_b), vals4 + vals8[:4] + vals8[4:]):
        ax.text(bar.get_x() + bar.get_width()/2, v - 25, f'{v:.0f}',
                ha='center', va='top', fontsize=7, fontweight='bold', color='white')

    # ── Plot 2: Summary bars ──
    ax = axes[1]
    systems  = ['4-Int\nCooperative', '8-Int\nGroup A', '8-Int\nGroup B', '8-Int\nOverall']
    averages = [avg4, ga_avg, gb_avg, np.mean(vals8)]
    errs     = [std4, std8, std8, std8]
    clrs     = [COLORS['network_4'], COLORS['group_a'], COLORS['group_b'], COLORS['network_8']]
    b = ax.bar(systems, averages, yerr=errs, capsize=5, color=clrs,
               alpha=0.85, edgecolor='black', linewidth=1.2,
               error_kw={'linewidth': 1.8})
    ax.set_title('Avg Reward per Intersection\nSystem-Level Comparison', fontweight='bold', fontsize=10)
    ax.set_ylabel('Avg Reward per Intersection')
    for bar, v in zip(b, averages):
        ax.text(bar.get_x() + bar.get_width()/2, v - 25, f'{v:.1f}',
                ha='center', va='top', fontsize=9, fontweight='bold', color='white')
    diff = np.mean(vals8) - avg4
    pct  = diff / abs(avg4) * 100
    ax.text(0.5, 0.03,
            f'Difference: {diff:+.1f} ({pct:+.1f}%)  {note}',
            transform=ax.transAxes, ha='center', fontsize=8.5,
            bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.9),
            fontweight='bold')

    # ── Plot 3: Network total reward (scaling) ──
    ax = axes[2]
    net_systems   = ['4-Int\nNetwork Total', '8-Int\nNetwork Total', '8-Int\n(Per-Int Scaled)']
    net_vals      = [net4, net8, avg4 * 8]
    net_clrs      = [COLORS['network_4'], COLORS['network_8'], '#95A5A6']
    b3 = ax.bar(net_systems, net_vals, color=net_clrs, alpha=0.85,
                edgecolor='black', linewidth=1.2)
    ax.set_title('Network Total Reward\nScaling: 4 → 8 Intersections', fontweight='bold', fontsize=10)
    ax.set_ylabel('Total Network Reward')
    for bar, v in zip(b3, net_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v - 60, f'{v:.0f}',
                ha='center', va='top', fontsize=10, fontweight='bold', color='white')
    ax.text(0.5, 0.03,
            'Expected: 8-Int total ≈ 2× 4-Int total (2 equal groups)',
            transform=ax.transAxes, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#D4EDDA', alpha=0.9))

    plt.suptitle('4-Intersection (4×1)  vs  8-Intersection (4×2 Grouped)\nPerformance Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = f'{OUTPUT_DIR}/performance_comparison.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
    print(f"✅ Saved: {out}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# Figure 4: Combined Summary (poster-style)
# ════════════════════════════════════════════════════════════════════
def fig_summary_poster(df4, eval8, is_projected):
    last50 = df4['tls1'].tail(50)
    avg4 = last50.mean()
    net4 = df4['network_reward'].tail(50).mean()

    if is_projected:
        per_int_vals = [avg4] * 8
        net8 = avg4 * 8
    else:
        per_int_vals = [eval8[f'tls_{i}'].mean() for i in range(1, 9)]
        net8 = eval8['network_reward'].mean()

    fig = plt.figure(figsize=(20, 12))
    gs  = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)
    fig.patch.set_facecolor('#1a1a2e')

    def styled_ax(ax, title=''):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#0f3460')
        ax.grid(True, color='#0f3460', linewidth=0.8)
        if title:
            ax.set_title(title, fontsize=10, fontweight='bold', color='white')

    # ── Row 0: training curves ──
    ax_curve = fig.add_subplot(gs[0, :3])
    styled_ax(ax_curve, '4-Intersection Training Progress (700 Episodes)')
    ax_curve.plot(df4['episode'], df4['network_reward'],
                  color=COLORS['network_4'], alpha=0.2, linewidth=1)
    ax_curve.plot(df4['episode'], df4['smooth_network'],
                  color=COLORS['network_4'], linewidth=2.5, label='4-Int network (smoothed)')
    ax_curve.axhline(net4, color='#E74C3C', linestyle='--', linewidth=1.5,
                     label=f'Final avg: {net4:.0f}')
    ax_curve.set_ylabel('Network Total Reward', color='white')
    ax_curve.set_xlabel('Episode', color='white')
    ax_curve.legend(fontsize=9)

    # ── Row 0: stats box ──
    ax_stats = fig.add_subplot(gs[0, 3])
    ax_stats.set_facecolor('#0f3460')
    ax_stats.axis('off')
    lines = [
        ('4-Int Cooperative', COLORS['network_4'], 'bold', 13),
        (f'Per-Int Final:  {avg4:+.1f}', 'white', 'normal', 11),
        (f'Net Total:       {net4:+.1f}', 'white', 'normal', 11),
        ('', '', '', 0),
        ('8-Int Grouped', COLORS['group_a'], 'bold', 13),
        (f'Group A avg:    {avg4:+.1f}', COLORS['group_a'], 'normal', 11),
        (f'Group B avg:    {avg4:+.1f}', COLORS['group_b'], 'normal', 11),
        (f'Net Total:       {net8:+.1f}', COLORS['network_8'], 'normal', 11),
        ('', '', '', 0),
        ('Stage 1 Status', '#F39C12', 'bold', 13),
        ('  Weights:  transferred ✓', '#aaaaaa', 'normal', 10),
        ('  Fine-tune: pending', '#aaaaaa', 'normal', 10),
        ('  GPU: RTX 2050', '#aaaaaa', 'normal', 10),
    ]
    y = 0.95
    for text, color, w, fs in lines:
        if not text:
            y -= 0.04; continue
        ax_stats.text(0.08, y, text, transform=ax_stats.transAxes,
                      color=color, fontweight=w, fontsize=fs, va='top')
        y -= 0.075

    # ── Row 1: per-intersection bars ──
    ax_per = fig.add_subplot(gs[1, :2])
    styled_ax(ax_per, 'Per-Intersection Reward — 4-Int vs 8-Int')
    x = np.arange(8)
    ax_per.bar(x[:4] - 0.2, [avg4]*4, 0.38, color=COLORS['network_4'],
               alpha=0.85, label='4-Int system')
    ax_per.bar(x[:4] + 0.2, per_int_vals[:4], 0.38, color=COLORS['group_a'],
               alpha=0.85, label='8-Int Group A')
    ax_per.bar(x[4:] + 0.2 - 4, per_int_vals[4:], 0.38, color=COLORS['group_b'],
               alpha=0.85, label='8-Int Group B')
    ax_per.set_xticks(x-0.2)
    ax_per.set_xticklabels([f'I{i}' for i in range(1, 9)])
    ax_per.set_ylabel('Avg Reward', color='white')
    ax_per.legend(fontsize=8.5)

    # ── Row 1: episode reward over training (last 50) ──
    ax_ep = fig.add_subplot(gs[1, 2:])
    styled_ax(ax_ep, 'Episode Reward — Final 50 Episodes (4-Int)')
    tail = df4.tail(50)
    ax_ep.bar(range(50), tail['network_reward'].values,
              color=COLORS['network_4'], alpha=0.7, edgecolor='none')
    ax_ep.axhline(net4, color='#E74C3C', linestyle='--', linewidth=1.5,
                  label=f'Mean: {net4:.0f}')
    ax_ep.set_xlabel('Last 50 Episodes', color='white')
    ax_ep.set_ylabel('Network Reward', color='white')
    ax_ep.legend(fontsize=9)

    # ── Row 2: scaling analysis ──
    ax_scale = fig.add_subplot(gs[2, :2])
    styled_ax(ax_scale, 'Network Scaling: Reward vs Number of Intersections')
    ints  = [4, 8]
    nets  = [net4, net8]
    scale = [net4/4, net8/8]
    ax_scale.plot(ints, nets, 'o-', color=COLORS['network_8'],
                  linewidth=2.5, markersize=9, label='Network Total')
    ax_twin = ax_scale.twinx()
    ax_twin.set_facecolor('#16213e')
    ax_twin.tick_params(colors='#45B7D1')
    ax_twin.plot(ints, scale, 's--', color=COLORS['group_a'],
                 linewidth=2, markersize=8, label='Avg per Intersection')
    ax_twin.set_ylabel('Per-Intersection Avg', color=COLORS['group_a'])
    ax_scale.set_xticks([4, 8])
    ax_scale.set_xlabel('Number of Intersections', color='white')
    ax_scale.set_ylabel('Total Network Reward', color=COLORS['network_8'])
    lines_a, labels_a = ax_scale.get_legend_handles_labels()
    lines_b, labels_b = ax_twin.get_legend_handles_labels()
    ax_scale.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8.5)

    # ── Row 2: group balance ──
    ax_bal = fig.add_subplot(gs[2, 2:])
    styled_ax(ax_bal, 'Intra-Group Reward Balance (8-Int Pre-FT)')
    ga_vals = per_int_vals[:4]
    gb_vals = per_int_vals[4:]
    tls_labels = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']
    ax_bal.bar(tls_labels[:4], ga_vals, color=COLORS['group_a'], alpha=0.85, label='Group A')
    ax_bal.bar(tls_labels[4:], gb_vals, color=COLORS['group_b'], alpha=0.85, label='Group B')
    ax_bal.axhline(np.mean(ga_vals), color=COLORS['group_a'], linestyle='--',
                   linewidth=1.5, alpha=0.7)
    ax_bal.axhline(np.mean(gb_vals), color=COLORS['group_b'], linestyle='--',
                   linewidth=1.5, alpha=0.7)
    ax_bal.set_ylabel('Avg Reward', color='white')
    ax_bal.legend(fontsize=9)

    proj_text = ' [PROJECTED — run eval after fine-tuning for real data]' if is_projected else ''
    plt.suptitle(
        f'Multi-Agent DDQN Traffic Control — 4×1 vs 4×2 Comparison{proj_text}',
        fontsize=15, fontweight='bold', color='white', y=1.01
    )
    out = f'{OUTPUT_DIR}/summary_poster.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"✅ Saved: {out}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nGenerating 4×1 vs 4×2 comparison visuals → {OUTPUT_DIR}/")
    print("─" * 60)

    df4 = load_4int_data()
    eval8, is_projected = load_8int_eval(df4)

    if is_projected:
        print("⚠  No 8-intersection eval data found → using projected values")
        print("   (Run evaluate_pretrain_8intersection.py to get real measurements)")
    else:
        print("✓  8-intersection eval data loaded from", DATA_8INT)

    print("\n[1/4] Architecture diagram...")
    fig_architecture()

    print("[2/4] Training curve analysis...")
    fig_training_curve(df4)

    print("[3/4] Performance comparison...")
    fig_performance_comparison(df4, eval8, is_projected)

    print("[4/4] Summary poster...")
    fig_summary_poster(df4, eval8, is_projected)

    print(f"\n✅ All 4 visuals saved to {OUTPUT_DIR}/")
    print("   architecture_comparison.png")
    print("   4int_training_analysis.png")
    print("   performance_comparison.png")
    print("   summary_poster.png")


if __name__ == '__main__':
    main()
