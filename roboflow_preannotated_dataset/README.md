# Roboflow Model-Assisted Pre-Annotated Dataset (Targeted Low-Confidence Mining)

This dataset contains **200 curated 640x640 microscopy crops** selected specifically for active-learning annotation in **Roboflow**.

### Dataset Composition
- **Total Crops:** 200
- **Targeted Low-Confidence Stomata Crops:** 185 (each crop centered on historically lowest-confidence or closed-pore stomata)
- **Background Distractor Crops:** 15 (cuticle folds, cell wall junctions, and hair scars with zero stomata)
- **Classes:**
  - `0`: `center` (pore aperture slit)
  - `1`: `stomata-huw1` (whole stoma / guard cell pair)

### Quick Roboflow Import Instructions
1. Upload `roboflow_preannotated_dataset.zip` directly to your Roboflow project.
2. Select format: **YOLOv8 Segmentation** or **YOLO11 Segmentation**.
3. All crops are pre-segmented: simply accept or adjust the boundaries in the Roboflow annotation UI.
