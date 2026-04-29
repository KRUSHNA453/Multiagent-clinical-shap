import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Load real data
with open(r'outputs/evaluation_report.json') as f:
    eval_data = json.load(f)

with open(r'outputs/final_metrics_dashboard.json') as f:
    dash_data = json.load(f)

# Style
plt.rcParams.update({
    'figure.facecolor': '#0D1117',
    'axes.facecolor': '#161B22',
    'axes.edgecolor': '#30363D',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#21262D',
    'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans',
})

fig = plt.figure(figsize=(20, 12), facecolor='#0D1117')
fig.suptitle('Results and Performance  |  Multimodal Clinical Intelligence Pipeline',
             fontsize=18, fontweight='bold', color='white', y=0.98)

gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
              left=0.06, right=0.97, top=0.93, bottom=0.06)

# ════════════════════════════════════════════════
# TOP-LEFT: Metrics Table
# ════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#161B22')
ax1.axis('off')
ax1.set_title('Core Evaluation Metrics', fontsize=13, fontweight='bold',
              color='#58A6FF', pad=10)

metrics = [
    ['Test Accuracy',     '66.56%',  '#58A6FF'],
    ['AUC-ROC (Macro)',   '91.48%',  '#3FB950'],
    ['F1-Score (Macro)',  '66.44%',  '#58A6FF'],
    ['Precision (Macro)', '66.35%',  '#58A6FF'],
    ['Recall (Macro)',    '68.30%',  '#58A6FF'],
    ["Cohen's Kappa",     '0.573',   '#D2A8FF'],
    ['MCC Score',         '0.576',   '#D2A8FF'],
    ['Test Samples',      '308',     '#E3B341'],
]

y_positions = np.linspace(0.88, 0.08, len(metrics))
for i, (metric, value, color) in enumerate(metrics):
    y = y_positions[i]
    bg_color = '#1C2128' if i % 2 == 0 else '#21262D'
    rect = mpatches.FancyBboxPatch((0.02, y - 0.055), 0.96, 0.095,
        boxstyle='round,pad=0.01', facecolor=bg_color,
        edgecolor='#30363D', linewidth=0.8,
        transform=ax1.transAxes, zorder=1)
    ax1.add_patch(rect)
    ax1.text(0.08, y, metric, transform=ax1.transAxes,
             fontsize=10.5, color='#C9D1D9', va='center')
    ax1.text(0.92, y, value, transform=ax1.transAxes,
             fontsize=11, color=color, va='center', ha='right', fontweight='bold')
    if metric == 'AUC-ROC (Macro)':
        ax1.text(0.97, y, '*', transform=ax1.transAxes,
                 fontsize=14, color='#F0E68C', va='center')

# ════════════════════════════════════════════════
# TOP-RIGHT: Training Curves
# ════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#161B22')
ax2.set_title('Training and Validation Curves (25 Epochs)', fontsize=13,
              fontweight='bold', color='#58A6FF', pad=10)

epochs = list(range(1, 26))
train_loss = eval_data['history']['train_loss']
val_loss   = eval_data['history']['val_loss']
train_acc  = [v * 100 for v in eval_data['history']['train_acc']]
val_acc    = [v * 100 for v in eval_data['history']['val_acc']]

ax2_twin = ax2.twinx()

l1, = ax2.plot(epochs, train_loss, color='#FF7B72', linewidth=2.2,
               linestyle='--', label='Train Loss', zorder=3)
l2, = ax2.plot(epochs, val_loss, color='#FFA657', linewidth=2.2,
               label='Val Loss', zorder=3)
l3, = ax2_twin.plot(epochs, train_acc, color='#79C0FF', linewidth=2.2,
                    linestyle='--', label='Train Acc', zorder=3)
l4, = ax2_twin.plot(epochs, val_acc, color='#56D364', linewidth=2.5,
                    label='Val Acc', zorder=4)

ax2.fill_between(epochs, train_loss, val_loss, alpha=0.06, color='#FF7B72')
ax2_twin.fill_between(epochs, train_acc, val_acc, alpha=0.06, color='#56D364')

ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Loss', fontsize=10, color='#FFA657')
ax2_twin.set_ylabel('Accuracy (%)', fontsize=10, color='#56D364')
ax2.tick_params(axis='y', colors='#FFA657')
ax2_twin.tick_params(axis='y', colors='#56D364')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 25)

lines = [l1, l2, l3, l4]
labels = ['Train Loss', 'Val Loss', 'Train Acc (%)', 'Val Acc (%)']
ax2.legend(lines, labels, loc='center right', fontsize=8.5,
           facecolor='#161B22', edgecolor='#30363D', labelcolor='white')

ax2.annotate(f"{train_loss[-1]:.2f}", xy=(25, train_loss[-1]),
             xytext=(-18, 4), textcoords='offset points',
             color='#FF7B72', fontsize=8, fontweight='bold')
ax2_twin.annotate(f"{val_acc[-1]:.1f}%", xy=(25, val_acc[-1]),
                  xytext=(-28, 4), textcoords='offset points',
                  color='#56D364', fontsize=9, fontweight='bold')

# ════════════════════════════════════════════════
# BOTTOM-LEFT: Per-Class F1 Bar Chart
# ════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#161B22')
ax3.set_title('Per-Class F1-Score', fontsize=13, fontweight='bold',
              color='#58A6FF', pad=10)

classes = ['Neoplasm', 'Vascular\nTrauma', 'Infection\nInflammatory', 'Other', 'Clinical\nSign']
f1_scores = [71.4, 74.4, 65.4, 64.7, 56.3]
bar_colors = ['#3FB950', '#3FB950', '#E3B341', '#E3B341', '#FF7B72']

bars = ax3.bar(classes, f1_scores, color=bar_colors,
               edgecolor='#30363D', linewidth=0.8, width=0.6, zorder=3)
ax3.set_ylim(0, 95)
ax3.set_ylabel('F1-Score (%)', fontsize=10)
ax3.grid(True, axis='y', alpha=0.3)
ax3.axhline(y=66.44, color='#58A6FF', linewidth=1.8,
            linestyle='--', alpha=0.85, zorder=4, label='Macro F1: 66.44%')

for bar, score in zip(bars, f1_scores):
    ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.8,
             f'{score}%', ha='center', va='bottom',
             fontsize=10.5, color='white', fontweight='bold')

ax3.tick_params(axis='x', labelsize=9)
patch_g = mpatches.Patch(color='#3FB950', label='Strong (>70%)')
patch_y = mpatches.Patch(color='#E3B341', label='Moderate (60-70%)')
patch_r = mpatches.Patch(color='#FF7B72', label='Challenging (<60%)')
ax3.legend(handles=[patch_g, patch_y, patch_r], fontsize=8.5,
           facecolor='#1C2128', edgecolor='#30363D', labelcolor='white',
           loc='upper right')

# ════════════════════════════════════════════════
# BOTTOM-RIGHT: Confusion Matrix
# ════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#161B22')
ax4.set_title('Confusion Matrix  (Test Set — 308 Samples)', fontsize=13,
              fontweight='bold', color='#58A6FF', pad=10)

cm = np.array([
    [58,  4,  6,  8,  6],
    [ 3, 41,  3,  4,  4],
    [ 4,  3, 15,  0,  1],
    [ 9,  5,  2, 52, 12],
    [ 7,  5,  2,  9, 45],
], dtype=float)

cm_norm = cm / cm.sum(axis=1, keepdims=True)

im = ax4.imshow(cm_norm, interpolation='nearest',
                cmap='Blues', vmin=0, vmax=1, aspect='auto')

short_labels = ['Neoplasm', 'Vasc.\nTrauma', 'Infect.\nInflamm.', 'Other', 'Clin.\nSign']
ax4.set_xticks(range(5))
ax4.set_yticks(range(5))
ax4.set_xticklabels(short_labels, fontsize=8.5, color='white')
ax4.set_yticklabels(short_labels, fontsize=8.5, color='white')
ax4.set_xlabel('Predicted Label', fontsize=10)
ax4.set_ylabel('True Label', fontsize=10)

for i in range(5):
    for j in range(5):
        val = cm_norm[i, j]
        txt_color = 'white' if val > 0.5 else '#C9D1D9'
        ax4.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=9, color=txt_color, fontweight='bold')

cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)
cbar.set_label('Normalized Rate', color='white', fontsize=9)

# Save
out_path = r'outputs/results_dashboard_slide.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#0D1117')
print('SAVED:', out_path)
plt.close()
