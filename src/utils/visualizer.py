"""
utils/visualizer.py
===================
Premium visualization utilities for Multimodal Clinical Intelligence Reports.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

def plot_consolidated_report(results, img, rag_df, case_id, project_root, save_filename="final_pipeline_demo.png"):
    """
    Generates a high-quality, professional multi-panel figure for the clinical report.
    
    Args:
        results: Dict containing 'gradcam_map', 'top_shap_tokens', 'final_report'.
        img: Original PIL Image.
        rag_df: Pandas DataFrame with similar cases.
        case_id: String identifier for the case.
        project_root: Root directory of the project for saving.
        save_filename: Name of the output image file.
    """
    # Set aesthetics
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    fig = plt.figure(figsize=(24, 22), dpi=150)
    gs = plt.GridSpec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.35, wspace=0.3)

    # ── 1. Original Image ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title("1. Original Patient Scan", fontsize=18, fontweight='bold', pad=15, color='#2c3e50')
    ax1.axis('off')
    # Add a thin border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('#bdc3c7')
        spine.set_linewidth(1)

    # ── 2. Grad-CAM Overlay ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cam_map = results.get('gradcam_map', np.zeros((224, 224)))
    
    # Robust image conversion for overlay
    if not isinstance(img, np.ndarray) and hasattr(img, 'resize'):
        img_arr = np.array(img.resize((224, 224)))
    else:
        # Ensure contiguous and correct dtype for cv2.resize
        img_np = np.ascontiguousarray(img)
        if img_np.dtype == np.float64 or img_np.dtype == np.float32:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        img_arr = cv2.resize(img_np, (224, 224))
    
    # Ensure img_arr is uint8 for cv2 operations
    if img_arr.max() <= 1.0:
        img_arr = (img_arr * 255).astype(np.uint8)
    else:
        img_arr = img_arr.astype(np.uint8)

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_arr, 0.6, heatmap, 0.4, 0)
    
    ax2.imshow(overlay)
    ax2.set_title("2. Grad-CAM Pathological Focus", fontsize=18, fontweight='bold', pad=15, color='#2c3e50')
    ax2.axis('off')

    # Add professional colorbar
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity', fontsize=12, labelpad=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])
    cbar.outline.set_edgecolor('#bdc3c7')

    # ── 3. SHAP Chart ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    shap_tokens = results.get('top_shap_tokens', [])[:12]
    tokens = [t['token'] for t in shap_tokens]
    scores = [t['shap_score'] for t in shap_tokens]

    # Sort by impact for better readability
    paired = sorted(zip(scores, tokens), 
                    key=lambda x: x[0], reverse=False)
    scores_sorted = [p[0] for p in paired]
    tokens_sorted = [p[1] for p in paired]

    # Use distinct colors for positive/negative impact
    colors = ['#e74c3c' if s > 0 else '#3498db' for s in scores_sorted]
    bars = ax3.barh(tokens_sorted, scores_sorted, color=colors, edgecolor='white', linewidth=0.8, height=0.7)

    # Add subtle grid
    ax3.grid(axis='x', linestyle='--', alpha=0.3)
    ax3.set_axisbelow(True)

    # Add value labels
    max_score = max(abs(s) for s in scores_sorted) if scores_sorted else 1.0
    for bar, score in zip(bars, scores_sorted):
        width = bar.get_width()
        offset = max_score * 0.05 if width >= 0 else -max_score * 0.15
        ax3.text(width + offset, bar.get_y() + bar.get_height()/2,
                 f'{score:+.4f}', va='center', fontsize=10, 
                 fontweight='bold', color='#34495e')

    ax3.set_title("3. SHAP Feature Attribution (Text)", fontsize=18, fontweight='bold', pad=15, color='#2c3e50')
    ax3.set_xlabel("Impact on Prediction (Shapley Value)", fontsize=13, labelpad=10)
    ax3.axvline(x=0, color='#2c3e50', linewidth=1.2, alpha=0.8)
    
    # Remove top/right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#bdc3c7')
    ax3.spines['bottom'].set_color('#bdc3c7')

    # Legend for SHAP
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Supports Predicted Class'),
        Patch(facecolor='#3498db', label='Contradicts Predicted Class')
    ]
    ax3.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True, shadow=True)

    # ── 4. RAG Table ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    if not rag_df.empty:
        rag_df['Diagnosis'] = rag_df['Diagnosis'].replace({
            'Inflammatory_Infection': 'Infection_Inflammatory',
            'inflammatory_infection': 'Infection_Inflammatory',
        })
        table_data = rag_df.values
        col_labels = rag_df.columns.tolist()

        table = ax4.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.1, 2.8)

        # Style header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#2c3e50')
            table[0, j].set_text_props(color='white', fontweight='bold', fontsize=13)
            table[0, j].set_edgecolor('#2c3e50')

        # Alternating row colors
        for i in range(1, len(table_data) + 1):
            facecolor = '#fdfefe' if i % 2 == 0 else '#f4f6f7'
            # Highlight top match
            if i == 1: facecolor = '#e8f8f5' 
            
            for j in range(len(col_labels)):
                table[i, j].set_facecolor(facecolor)
                table[i, j].set_edgecolor('#d5dbdb')
                if i == 1: table[i, j].set_text_props(fontweight='bold', color='#16a085')

    ax4.set_title("4. Retrieval-Augmented Context (Top Cases)", fontsize=18, fontweight='bold', pad=15, color='#2c3e50')

    # ── 5. Clinical Report Text ────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    report_text = results.get('final_report', "No report generated.")
    # Clean markdown
    import re
    def strip_emojis(text):
        return re.sub(r'[^\x00-\x7F]+', ' ', text).strip()

    clean_report = strip_emojis(report_text)
    clean_report = clean_report.replace('##', '\n\n')
    clean_report = clean_report.replace('**', '')
    # Limit for visualization clarity
    words = clean_report.split()
    truncated = ' '.join(words[:450])
    if len(words) > 450: truncated += ' ... [Refer to full text for details]'

    ax5.text(
        0.01, 0.97, truncated,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='left',
        transform=ax5.transAxes,
        linespacing=1.8,
        bbox=dict(
            boxstyle='round,pad=1.0',
            facecolor='#f8f9fa',
            edgecolor='#2c3e50',
            linewidth=1.5,
            alpha=0.95
        )
    )
    ax5.set_title("5. Multi-Agent Synthetic Clinical Report", 
                  fontsize=18, fontweight='bold', loc='left', pad=20, color='#2c3e50')

    # ── Main Title ─────────────────────────────────────────────
    fig.patch.set_facecolor('#fdfefe')
    plt.suptitle(
        f"MULTIMODAL CLINICAL INTELLIGENCE REPORT\nCASE ID: {case_id}",
        fontsize=26,
        fontweight='bold',
        y=0.98,
        color='#1a252f'
    )

    # ── Footer ─────────────────────────────────────────────────
    fig.text(0.5, 0.01,
        "This report is AI-generated and intended for clinical "
        "decision support. Final diagnosis requires expert "
        "medical validation.",
        ha='center', fontsize=10, style='italic',
        color='#555555', alpha=0.6)

    # ── Save ───────────────────────────────────────────────────
    save_path = os.path.join(project_root, "outputs", save_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
    plt.show()
    print(f"SUCCESS: Consolidated Clinical Report saved to: {save_path}")
