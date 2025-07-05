# 🛍️ RecSys Challenge 2015 — Purchase Prediction

## 🧠 Overview

This repository contains my solution to the **RecSys Challenge 2015**, based on the YooChoose dataset. The objective was to predict the next item a user would interact with, given their prior behavior.

My approach focused on comparing three modeling strategies — **Gradient Boosting Machines (GBMs)**, **Feedforward Neural Networks (FNNs)**, and **Linear Regression** applied to a sparse, engineered feature set. After extensive experimentation, the most effective method was a **two-phase GBM strategy**:
1. **Phase 1**: Predict whether a session will lead to a purchase.
2. **Phase 2**: Predict which item will be purchased in high-probability sessions.

This two-step pipeline achieved **10.71% of the maximum possible score**, demonstrating strong performance despite the sparse nature of the data.

## 📁 Folder Structure

- `report.md` — A comprehensive report covering the methodology, model comparison, and key insights.
- `src/` — Source code for training, analysis, and utilities:
  - `fnn_model.py` — Implementation of the Feedforward Neural Network.
  - `utils.py` — Helper functions, including the challenges evaluation measure.
  - `eda.ipynb` — Exploratory Data Analysis of the YooChoose dataset.
  - `analysis.ipynb` — Feature engineering and analysis workflow.
  - `train.ipynb` — Training and evaluation of different models.

## 🚀 Highlights

- ✨ Two-stage GBM model pipeline for sparse purchase prediction
- 🧪 Comparative analysis of multiple ML approaches
- 📊 Clean and modular notebooks for EDA and training
- 🛠️ Custom evaluation metric aligned with challenge scoring
