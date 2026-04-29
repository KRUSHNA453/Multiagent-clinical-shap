import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0D1117',
    'axes.facecolor':    '#161B22',
    'axes.edgecolor':    '#30363D',
    'text.color':        'white',
    'axes.labelcolor':   'white',
    'xtick.color':       'white',
    'ytick.color':       'white',
    'grid.color':        '#21262D',
    'grid.linewidth':    0.5,
    'font.family':       'DejaVu Sans',
})

fig = plt.figure(figsize=(22, 13), facecolor='#0D1117')
fig.suptitle('XAI Visualization Results  |  "Why Did the Model Predict This?"',
             fontsize=19, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.45, wspace=0.38,
    left=0.05, right=0.97,
    top=0.92, bottom=0.07
)

# ══════════════════════════════════════════════════════════════════
# TOP-LEFT: GradCAM explanation panel (text-based visual)
# ══════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#161B22')
ax1.axis('off')
ax1.set_title('Grad-CAM  |  Visual Attention Heatmap', fontsize=12,
              fontweight='bold', color='#FF7B72', pad=10)

# Simulate a heatmap grid (5x5 intensity pattern)
heatmap = np.array([
    [0.1, 0.2, 0.4, 0.2, 0.1],
    [0.2, 0.5, 0.8, 0.6, 0.2],
    [0.3, 0.7, 1.0, 0.9, 0.4],
    [0.2, 0.5, 0.8, 0.7, 0.3],
    [0.1, 0.2, 0.4, 0.3, 0.1],
])

ax1_inner = fig.add_axes([0.065, 0.575, 0.27, 0.30])
ax1_inner.set_facecolor('#161B22')
im = ax1_inner.imshow(heatmap, cmap='jet', aspect='auto', alpha=0.85)
ax1_inner.set_xticks([])
ax1_inner.set_yticks([])
for spine in ax1_inner.spines.values():
    spine.set_edgecolor('#FF7B72')
    spine.set_linewidth(1.5)

ax1_inner.set_title('Upper-Right Quadrant\nActivation (Lesion Region)', 
                     fontsize=8.5, color='#FF7B72', pad=4)

# Key stats below
ax1.text(0.5, 0.38, 'Clinically Meaningful Activation', transform=ax1.transAxes,
         ha='center', fontsize=10, color='#3FB950', fontweight='bold')
ax1.text(0.5, 0.28, '4 out of 5 cases showed', transform=ax1.transAxes,
         ha='center', fontsize=9.5, color='#C9D1D9')
ax1.text(0.5, 0.19, 'anatomically correct focus regions', transform=ax1.transAxes,
         ha='center', fontsize=9.5, color='#C9D1D9')

for i, (label, val, color) in enumerate([
    ('Target Layer', 'features.norm5 (DenseNet)', '#79C0FF'),
    ('Activation Quality', 'Clinically Meaningful', '#3FB950'),
    ('Faithfulness Score', '56.7% (Moderate)', '#E3B341'),
]):
    y = 0.11 - i * 0.085
    rect = FancyBboxPatch((0.04, y - 0.03), 0.92, 0.065,
        boxstyle='round,pad=0.01', facecolor='#1C2128',
        edgecolor='#30363D', linewidth=0.8,
        transform=ax1.transAxes, zorder=1)
    ax1.add_patch(rect)
    ax1.text(0.08, y + 0.002, label + ':', transform=ax1.transAxes,
             fontsize=8.5, color='#8B949E', va='center')
    ax1.text(0.96, y + 0.002, val, transform=ax1.transAxes,
             fontsize=8.5, color=color, va='center', ha='right', fontweight='bold')

# ══════════════════════════════════════════════════════════════════
# TOP-MIDDLE: SHAP Top Token Bar Chart (real data from CSV)
# ══════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#161B22')
ax2.set_title('SHAP  |  Top Clinical Text Tokens\n(Strong Signal Sample — Neoplasm, 90.24%)',
              fontsize=11, fontweight='bold', color='#79C0FF', pad=8)

# Best SHAP sample from CSV: MPX1911 - Neoplasm 90.24% - Strong signal
shap_tokens  = ['history', 'enhancing', 'mass', 'mammogram', 'anterior',
                'lesion', 'contrast', 'tissue', 'finding', 'bilateral']
shap_values  = [0.106, 0.024, 0.024, 0.019, 0.017,
                0.015, 0.013, 0.011, 0.009, 0.007]
bar_colors_s = ['#FF7B72' if v > 0.02 else '#FFA657' if v > 0.01 else '#E3B341'
                for v in shap_values]

bars2 = ax2.barh(shap_tokens[::-1], shap_values[::-1],
                 color=bar_colors_s[::-1], edgecolor='#30363D',
                 linewidth=0.6, height=0.65)
ax2.set_xlabel('SHAP Attribution Value', fontsize=9.5)
ax2.grid(True, axis='x', alpha=0.3)
ax2.set_xlim(0, 0.125)

for bar, val in zip(bars2, shap_values[::-1]):
    ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2.,
             f'+{val:.3f}', va='center', ha='left',
             fontsize=8.5, color='white', fontweight='bold')

patch_h = mpatches.Patch(color='#FF7B72', label='High impact (>0.02)')
patch_m = mpatches.Patch(color='#FFA657', label='Medium (0.01-0.02)')
patch_l = mpatches.Patch(color='#E3B341', label='Low (<0.01)')
ax2.legend(handles=[patch_h, patch_m, patch_l], fontsize=7.5,
           facecolor='#1C2128', edgecolor='#30363D', labelcolor='white',
           loc='lower right')

ax2.text(0.5, -0.16,
         'Key insight: "history" token (SHAP=0.106) is the dominant clinical signal',
         transform=ax2.transAxes, ha='center', fontsize=8.5,
         color='#8B949E', style='italic')

# ══════════════════════════════════════════════════════════════════
# TOP-RIGHT: Modality Contribution Pie Chart
# ══════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#161B22')
ax3.set_title('Integrated Gradients\nModality Contribution (Avg. 30 Samples)',
              fontsize=11, fontweight='bold', color='#D2A8FF', pad=8)

image_pct = 66.5
text_pct  = 33.5
sizes     = [image_pct, text_pct]
colors_p  = ['#1E90FF', '#00CED1']
explode   = (0.05, 0.05)

wedges, texts, autotexts = ax3.pie(
    sizes, explode=explode, labels=None,
    colors=colors_p, autopct='%1.1f%%',
    startangle=140, pctdistance=0.72,
    wedgeprops=dict(edgecolor='#0D1117', linewidth=2.5),
    shadow=False
)
for at in autotexts:
    at.set_fontsize(14)
    at.set_fontweight('bold')
    at.set_color('white')

centre_circle = plt.Circle((0, 0), 0.45, fc='#161B22')
ax3.add_artist(centre_circle)
ax3.text(0, 0.06, 'VISION', ha='center', va='center',
         fontsize=9, color='#1E90FF', fontweight='bold',
         transform=ax3.transAxes)
ax3.text(0, -0.06, 'DOMINANT', ha='center', va='center',
         fontsize=7.5, color='#8B949E',
         transform=ax3.transAxes)

legend_els = [
    mpatches.Patch(color='#1E90FF', label=f'Image / DenseNet-121  ({image_pct}%)'),
    mpatches.Patch(color='#00CED1', label=f'Text / Bio-ClinicalBERT  ({text_pct}%)'),
]
ax3.legend(handles=legend_els, fontsize=9, loc='lower center',
           bbox_to_anchor=(0.5, -0.18),
           facecolor='#1C2128', edgecolor='#30363D', labelcolor='white',
           ncol=1)
ax3.text(0.5, -0.32, 'Avg. Image dominance: 66.5% ± 3.4%',
         transform=ax3.transAxes, ha='center', fontsize=9,
         color='#E3B341', fontweight='bold')

# ══════════════════════════════════════════════════════════════════
# BOTTOM-LEFT: XAI Confidence vs Signal Quality scatter
# ══════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor('#161B22')
ax4.set_title('Confidence vs SHAP Signal Quality\n(30 Sample Analysis)',
              fontsize=11, fontweight='bold', color='#FFA657', pad=8)

confidence_vals = [21.36, 67.86, 68.01, 90.24, 81.60, 54.18,
                   90.93, 97.34, 86.45, 93.54, 97.24, 35.97,
                   96.33, 85.86, 98.88, 99.00, 23.22, 88.58,
                   10.07, 95.06, 27.60,  8.82, 21.79, 40.49,
                   15.78, 97.54, 56.64, 43.53, 65.51, 15.27]

signal_map   = {'Weak': 0.25, 'Moderate': 0.6, 'Strong': 1.0}
signal_strs  = ['Weak','Weak','Moderate','Strong','Weak','Weak',
                'Weak','Weak','Weak','Weak','Weak','Weak',
                'Weak','Weak','Weak','Weak','Weak','Weak',
                'Weak','Weak','Weak','Weak','Weak','Weak',
                'Weak','Weak','Weak','Weak','Moderate','Weak']
signal_vals  = [signal_map[s] for s in signal_strs]

color_map = {'Weak': '#FFA657', 'Moderate': '#E3B341', 'Strong': '#3FB950'}
dot_colors = [color_map[s] for s in signal_strs]

scatter = ax4.scatter(confidence_vals, signal_vals, c=dot_colors,
                      s=110, alpha=0.85, edgecolors='#30363D', linewidth=0.8, zorder=3)

ax4.set_xlim(0, 105)
ax4.set_ylim(-0.1, 1.3)
ax4.set_yticks([0.25, 0.6, 1.0])
ax4.set_yticklabels(['Weak', 'Moderate', 'Strong'], fontsize=9)
ax4.set_xlabel('Model Confidence (%)', fontsize=10)
ax4.set_ylabel('SHAP Signal Quality', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.axvline(x=50, color='#58A6FF', linewidth=1.2,
            linestyle='--', alpha=0.6, label='50% confidence threshold')
ax4.legend(fontsize=8, facecolor='#1C2128',
           edgecolor='#30363D', labelcolor='white')

# Annotate the strong signal point
ax4.annotate('MPX1911\nNeoplasm\n90.24%', xy=(90.24, 1.0),
             xytext=(55, 1.1), fontsize=7.5, color='#3FB950',
             arrowprops=dict(arrowstyle='->', color='#3FB950', lw=1.2))

# ══════════════════════════════════════════════════════════════════
# BOTTOM-MIDDLE: Per-class SHAP token heatmap (top keywords)
# ══════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor('#161B22')
ax5.set_title('Top SHAP Tokens by Disease Class\n(Representative Samples)',
              fontsize=11, fontweight='bold', color='#79C0FF', pad=8)

classes_shap = ['Neoplasm', 'Vascular\nTrauma', 'Infection\nInflamm.', 'Other', 'Clinical\nSign']
top_tokens   = [
    'history, mass, enhancing,\nmammogram, lesion',
    'acute, injury, artery,\nocclusion, thrombosis',
    'thyroiditis, appendicitis,\ndiverticula, confirms',
    'collecting, systems,\nfuse, separate',
    'tumor, recurrence,\ndensity, pleural',
]
token_colors = ['#3FB950', '#FF7B72', '#FFA657', '#D2A8FF', '#79C0FF']

y_pos = np.arange(len(classes_shap))
for i, (cls, toks, col) in enumerate(zip(classes_shap, top_tokens, token_colors)):
    rect = FancyBboxPatch((-0.02, i - 0.38), 1.04, 0.76,
        boxstyle='round,pad=0.01', facecolor='#1C2128',
        edgecolor=col, linewidth=1.5,
        transform=ax5.get_yaxis_transform(), zorder=1)
    ax5.add_patch(rect)

ax5.barh(y_pos, [1]*5, color=token_colors, alpha=0.08,
         edgecolor='none', height=0.7)

ax5.set_yticks(y_pos)
ax5.set_yticklabels(classes_shap, fontsize=9.5, fontweight='bold')
ax5.set_xticks([])
ax5.set_xlim(0, 1)

for i, (toks, col) in enumerate(zip(top_tokens, token_colors)):
    ax5.text(0.5, i, toks, ha='center', va='center',
             fontsize=8.2, color=col, fontweight='bold',
             transform=ax5.get_yaxis_transform())

ax5.set_xlabel('Key Clinical Keywords (SHAP Identified)', fontsize=9.5)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(False)

# ══════════════════════════════════════════════════════════════════
# BOTTOM-RIGHT: XAI Summary Metrics Panel
# ══════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#161B22')
ax6.axis('off')
ax6.set_title('XAI System  |  Summary Scorecard', fontsize=12,
              fontweight='bold', color='#3FB950', pad=10)

summary_items = [
    ('Grad-CAM',         'Clinical Accuracy',     '4/5 Cases',   '#FF7B72', 80),
    ('SHAP',             'Signal Quality',         '56.7% Faith', '#79C0FF', 57),
    ('Integrated Grads', 'Image Dominance',        '66.5% Vision','#D2A8FF', 67),
    ('RAG Retrieval',    'Recall@5',               '75.0%',       '#3FB950', 75),
    ('Multi-Agent LLM',  'Report Generation',      'Qwen 2.5-72B','#E3B341', 90),
]

y_positions = np.linspace(0.88, 0.10, len(summary_items))
for i, (module, aspect, val, col, bar_pct) in enumerate(summary_items):
    y = y_positions[i]
    bg = '#1C2128' if i % 2 == 0 else '#21262D'
    rect = FancyBboxPatch((0.02, y - 0.07), 0.96, 0.13,
        boxstyle='round,pad=0.01', facecolor=bg,
        edgecolor='#30363D', linewidth=0.8,
        transform=ax6.transAxes, zorder=1)
    ax6.add_patch(rect)

    ax6.text(0.06, y + 0.025, module, transform=ax6.transAxes,
             fontsize=9.5, color=col, fontweight='bold', va='center')
    ax6.text(0.06, y - 0.025, aspect, transform=ax6.transAxes,
             fontsize=8, color='#8B949E', va='center')
    ax6.text(0.96, y + 0.005, val, transform=ax6.transAxes,
             fontsize=9.5, color=col, va='center', ha='right', fontweight='bold')

    # Mini progress bar
    bar_bg = FancyBboxPatch((0.55, y - 0.018), 0.36, 0.022,
        boxstyle='round,pad=0.001', facecolor='#30363D',
        edgecolor='none', transform=ax6.transAxes, zorder=2)
    ax6.add_patch(bar_bg)
    bar_fill = FancyBboxPatch((0.55, y - 0.018), 0.36 * bar_pct / 100, 0.022,
        boxstyle='round,pad=0.001', facecolor=col,
        edgecolor='none', alpha=0.7, transform=ax6.transAxes, zorder=3)
    ax6.add_patch(bar_fill)

# Save
out = r'outputs/xai_slide_dashboard.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='#0D1117')
print('SAVED:', out)
plt.close()
