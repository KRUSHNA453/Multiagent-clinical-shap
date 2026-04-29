import nbformat as nbf
import os

# Path to the notebook
notebook_path = r'z:\study files\SRM_study\SEM-2\AML-509_Agentic AI and GAN\Team Project\Agent_code_Trial-2\Trial_2\notebooks\07_full_pipeline_demo_executed.ipynb'

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Define the new markdown cell
table_markdown = """### Clinical Classification Reference
The following table describes the diagnostic groups used in this pipeline:

| Id | Group | Description |
| :-- | :--- | :--- |
| 0 | Neoplasm | Refers to the abnormal and excessive growth of tissue, which may be benign (non-cancerous) or malignant (cancerous) in nature. |
| 1 | Vascular_Trauma | Encompasses conditions involving injury or damage to blood vessels, as well as complications arising from physical trauma to the circulatory system. |
| 2 | Infection_Inflammatory | Includes diseases caused by invading pathogens (bacteria, viruses, etc.) and conditions characterized by the body's inflammatory response. |
| 3 | Other | A general category for miscellaneous medical findings, congenital anomalies, or diagnoses that do not fall within the specific defined groups. |
| 4 | Clinical Sign | Pertains to objective medical evidence or physical manifestations of a condition observed by a healthcare professional during an examination. |"""

new_cell = nbf.v4.new_markdown_cell(table_markdown)

# Insert the new cell after the first cell (index 1)
nb.cells.insert(1, new_cell)

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully inserted the medical classification table into the notebook.")
