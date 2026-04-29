import json
import os

path = r'z:\study files\SRM_study\SEM-2\AML-509_Agentic AI and GAN\Team Project\Agent_code_Trial-2\Trial_2\notebooks\02_preprocessing.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that loads the dataset
found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('medpix_master.csv' in line for line in cell.get('source', [])):
        # Insert a new cell right after this one
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['diagnosis'] = df['diagnosis'].str.strip()\n",
                "df['diagnosis'] = df['diagnosis'].replace({\n",
                "    'Inflammatory_Infection': 'Infection_Inflammatory',\n",
                "    'inflammatory_infection': 'Infection_Inflammatory',\n",
                "    'infection_inflammatory': 'Infection_Inflammatory',\n",
                "})\n",
                "print(\"Unique labels:\", sorted(df['diagnosis'].unique()))\n"
            ]
        }
        nb['cells'].insert(i+1, new_cell)
        found = True
        break

if found:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Notebook 02 updated successfully.')
else:
    print('Dataset loading cell not found.')
