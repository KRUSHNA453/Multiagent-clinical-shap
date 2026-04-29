import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, '.')

from src.utils.visualizer import plot_consolidated_report

# Mock data
case_id = 'MOCK_CASE_001'
img = torch.randn(3, 224, 224)
results = {
    'predicted_label': 'Infection_Inflammatory',
    'confidence': 0.954321,
    'shap_tokens': [('token1', 0.1), ('token2', -0.05), ('token3', 0.2)],
    'gradcam_heatmap': np.random.rand(224, 224),
    'final_report': 'This is a mock report with an emoji 💊. It should be stripped.'
}
rag_df = pd.DataFrame({
    'Case ID': ['CASE1', 'CASE2'],
    'Diagnosis': ['Infection_Inflammatory', 'Neoplasm'],
    'Similarity': [0.85, 0.75]
})

try:
    # Permute to (H, W, C) for matplotlib
    img_permuted = img.permute(1, 2, 0).numpy()
    plot_consolidated_report(results, img_permuted, rag_df, case_id, project_root='.')
    print("DONE: plot_consolidated_report executed successfully.")
    
    # Check output file
    if os.path.exists('outputs/final_pipeline_demo.png'):
        size = os.path.getsize('outputs/final_pipeline_demo.png')
        print(f"SUCCESS: Output file created: outputs/final_pipeline_demo.png ({size/1024:.2f} KB)")
    else:
        print("FAILED: Output file NOT found.")
except Exception as e:
    print(f"ERROR: Execution failed: {e}")
