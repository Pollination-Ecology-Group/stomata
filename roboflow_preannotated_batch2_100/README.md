# Roboflow Model-Assisted Pre-Annotated Dataset — Batch 2 (100 Hard-Case Crops)

This dataset contains **100 curated 640x640 microscopy crops** (numbered **201 to 300**) selected specifically to boost model performance on previously challenging edge cases.

### Dataset Composition
- **Total Crops in Batch 2:** 100 (Crops 201 to 300)
- **Targeted Hard-Case Stomata Crops:** 90
  - **Neopolyploids (`neo_4x`):** 28 crops (dramatically expanding representation from Batch 1)
  - **Natural Tetraploids (`nat_4x`):** 26 crops (large guard cell pairs, irregular boundaries)
  - **Natural Diploids (`nat_2x`):** 18 crops
  - **Synthetic Diploids (`ory_2x`):** 18 crops
- **Leaf Surface Distribution:**
  - **Adaxial (Upper Surface):** 46 crops (challenging jigsaw pavement cell ridges & thick cuticles)
  - **Abaxial (Lower Surface):** 44 crops (dense clusters & focal blur gradients)
- **Hard Negative Background Distractor Crops:** 10 (empty label files with zero stomata)
- **Classes:**
  - `0`: `center` (pore aperture slit)
  - `1`: `stomata-huw1` (whole stoma / guard cell pair)

### Quick Roboflow Import Instructions
1. Upload `roboflow_preannotated_batch2_100.zip` directly to your Roboflow project.
2. Select format: **YOLOv8 Segmentation** or **YOLO11 Segmentation**.
3. All crops are pre-segmented: review and fine-tune polygon boundaries in Roboflow.
