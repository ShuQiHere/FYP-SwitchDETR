# FYP-SwitchDETR

This repository contains the official implementation of **Switch-DETR** (also referred to as Switch-Net in the codebase), a state-of-the-art model for video moment retrieval and highlight detection.

## Overview

Switch-DETR addresses the "query-FFN specialization collapse" in traditional DETR-based video moment retrieval models. By replacing the shared Feed-Forward Network (FFN) in the decoder with a **sparse Mixture-of-Experts (MoE) FFN**, Switch-DETR allows queries to specialize effectively, significantly improving ranking performance (R@1) and mean Average Precision (mAP).

## Key Features

*   **Switch-DETR Architecture**: An expert decoder refinement network that equips every transformer decoder layer with a lightweight Top-2 MoE FFN.
*   **Distill Align Module**: Solves overlapping semantic information in contrastive learning.
*   **Convolutional Fuser**: Efficiently extracts local video features.
*   **Loop Decoder**: Iteratively refines results with a specialized MoE mechanism.

## Data Preparation

To replicate the results, you need to download the pre-extracted feature files for the datasets and organize them correctly.

### 1. QVHighlights
Download the official feature files for the QVHighlights dataset from `moment_detr_features.tar.gz` (8GB).

```bash
tar -xf path/to/moment_detr_features.tar.gz
```

If the official link is inaccessible, you can download the features from the backup link:
*   **QVHighlights (9.34GB)**: [Download](https://drive.google.com/file/d/1LXsZZBsv6Xbg_MmNQOezw0QYKccjcOkP/view)

### 2. Other Datasets
We provide extracted features for other supported datasets:

*   **Charades-STA (33.18GB)** (Including SlowFast+CLIP and VGG features): [Download](https://drive.google.com/file/d/1B2721QC799qbbGLGSa7DkXJjdRefvZf-/view)
*   **TACoS (290.7MB)**: [Download](https://drive.google.com/file/d/10Ji9MrlDK_4FdD3HotrVc407xVr4arsL/view)
*   **TVSum (69.1MB)**: *(Link provided upon request)*

### 3. Project Directory Structure
After downloading and extracting the features, please organize your project directory as follows. Alternatively, you can modify the paths in `switch-net.ipynb` to match your local setup.

```text
FYP-SwitchDETR/
├── switch-net.ipynb
├── README.md
├── qvhighlights/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── features/
│       ├── video_clip_features/
│       └── query_clip_features/
├── charades/
│   └── ...
└── ...
```

## Requirements

The code relies on standard deep learning libraries:
*   Python 3.x
*   PyTorch
*   NumPy
*   SciPy
*   Tqdm
*   WandB (for logging)

## Usage

The entire model logic and training pipeline are encapsulated in the provided Jupyter Notebook (`switch-net.ipynb`). You can run the notebook directly to explore the architecture or train the model on your dataset.

## Reference

If you use this code in your research, please refer to the associated thesis/paper:

> **Switch-DETR: Expert Decoder Refinement for Video Moment Retrieval**
