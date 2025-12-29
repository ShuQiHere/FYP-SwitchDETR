# FYP-SwitchDETR

This repository contains the official implementation of **Switch-DETR** (also referred to as Switch-Net in the codebase), a state-of-the-art model for video moment retrieval and highlight detection.

## Overview

Switch-DETR addresses the "query-FFN specialization collapse" in traditional DETR-based video moment retrieval models. By replacing the shared Feed-Forward Network (FFN) in the decoder with a **sparse Mixture-of-Experts (MoE) FFN**, Switch-DETR allows queries to specialize effectively, significantly improving ranking performance (R@1) and mean Average Precision (mAP).

## Key Features

*   **Switch-DETR Architecture**: An expert decoder refinement network that equips every transformer decoder layer with a lightweight Top-2 MoE FFN.
*   **Distill Align Module**: Solves overlapping semantic information in contrastive learning.
*   **Convolutional Fuser**: Efficiently extracts local video features.
*   **Loop Decoder**: Iteratively refines results with a specialized MoE mechanism.

## File Structure

*   `switch-net.ipynb`: The main notebook containing the complete source code for the model, including:
    *   **Data Loading & Preprocessing**
    *   **Model Architecture** (Unimodal Encoder, Distill Align, Convolutional Fuser, Switch-DETR Decoder with MoE)
    *   **Loss Functions** (Span Loss, Saliency Loss, Load Balancing Loss)
    *   **Training Loop & Validation**

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
