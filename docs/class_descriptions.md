# Clinical Classification Reference Guide

This document defines the diagnostic categories used in the Multimodal Clinical Intelligence Pipeline.

| Class ID | Class Name | Clinical Description |
| :--- | :--- | :--- |
| 0 | **Neoplasm** | Abnormal growth of tissue, which if cancer, is malignant. Includes primary tumors and metastatic disease. |
| 1 | **Vascular_Trauma** | Issues related to blood vessels (e.g., aneurysms, stenosis) or physical injury/trauma to anatomical structures. |
| 2 | **Infection_Inflammatory** | Conditions caused by pathogens (bacteria, viruses, fungi) or the body's immune/inflammatory response (e.g., autoimmune disorders). |
| 3 | **Other** | Miscellaneous conditions not fitting the main categories, including metabolic disorders or congenital anomalies. |
| 4 | **Clinical Sign** | Objective medical facts or characteristics detected by a physician during an examination (e.g., radiological signs). |

## Label Normalization Note
As of the latest update, all references to the inflammatory category have been standardized to `Infection_Inflammatory`. Legacy code or datasets using `Inflammatory_Infection` should be updated to this canonical form.
