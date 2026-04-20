# skin-analysis
# RF-DETR Training Pipeline with COCO Dataset + AWS SageMaker

This repository provides an end-to-end pipeline for:
- Merging multiple COCO datasets
- Splitting datasets into train/validation/test sets
- Training RF-DETR models using AWS SageMaker
- Running inference on trained models

---

## Project Overview

The project consists of three main components:

1. **Dataset Preparation**
   - Combine multiple COCO datasets
   - Normalize categories and annotations
   - Split into train / valid / test sets

2. **Model Training (SageMaker)**
   - Upload training code to S3
   - Launch SageMaker training job
   - Train RF-DETR models on GPU instances
   - Log experiments using MLflow

3. **Inference Pipeline**
   - Load trained checkpoint
   - Run inference on images
   - Save annotated outputs

---

## Project Files
  -dataset_utils.py # COCO merge + split logic
  
  -train_sagemaker.py # SageMaker training job launcher
  
  -train.py # RF-DETR training script (SageMaker entry point)
  
  -inference.py # Inference on trained model
  
  -requirements.txt # Python dependencies
  
  -README.md # Documentation

---

## Requirements

numpy==1.26.4
torch==2.2.2
ultralytics==8.4.13
mlflow==3.9.0
scikit-learn==1.8.0
optuna==4.7.0
boto3==1.42.46
sagemaker==3.4.1
pycocotools==2.0.11

---

## MLflow Tracking

During training, the following are logged:

  -Training loss

  -Evaluation metrics (if available)

---

## Notes
-Dataset must strictly follow COCO format

-Image filenames must match annotations

-GPU instances recommended for training

-Ensure S3 dataset path is correct and accessible

-Model checkpoints

-Artifacts (plots, JSON logs)


