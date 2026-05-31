# Power Quality Fault Detection

A PyTorch-based machine learning project for **multiclass power quality fault detection** using engineered electrical features such as RMS voltage, THD, DWT energies, entropy, and signal-to-noise ratio.

## Project Overview

This project focuses on classifying power quality conditions into multiple fault categories using tabular signal-derived features.

The workflow includes:

- Downloading the dataset from Kaggle
- Exploring class balance and feature relationships
- Encoding categorical variables
- Training a PyTorch-based neural network classifier
- Evaluating model performance using training curves and a confusion matrix

This project demonstrates how feature-engineered electrical measurements can be used to automatically detect and classify power quality disturbances.

## Dataset

The notebook uses the Kaggle dataset:

**Power Quality Fault Detection Dataset**

The dataset is loaded from:

```python
power_quality_fault_dataset.csv
```

### Available features

The model uses the following input features:

- Phase
- RMS_Voltage
- Peak_Voltage
- THD
- Duration_ms
- DWT_Energy_Level1
- DWT_Energy_Level2
- DWT_Entropy
- Signal_Noise_Ratio_dB

### Target

The target column is:

- `Label`

The notebook works on a dataset with **2367 samples** and a multiclass target representing different power quality conditions.

## Data Exploration

### Class Distribution

![Class Distribution](images/class_dist.png)

### Feature Correlation Heatmap

![Feature Heatmap](images/feature_heatmap.png)

These visualizations help understand dataset balance and relationships between the numerical electrical features.

## Model

The project is implemented in **PyTorch** using a neural-network-based classifier for tabular data.

Main libraries used:

- `torch`
- `torch.nn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

The model learns from electrical quality indicators rather than raw waveform images, making this a clean feature-based fault classification study.

## Training

The notebook includes a full training workflow with:

- preprocessing and feature preparation,
- train-validation split,
- model training,
- loss and accuracy tracking,
- and final evaluation.

### Accuracy Curve

![Accuracy Curve](images/accuracy.png)

### Loss Curve

![Loss Curve](images/loss.png)

## Evaluation

### Confusion Matrix

![Confusion Matrix](images/conf_matrix.png)

The confusion matrix shows how well the model distinguishes between the different power quality fault classes.

## Repository Structure

```bash
power_quality_fault_detection/
│── 2.-power-quality-fault-detection.ipynb
│── README.md
│── requirements.txt
│── .gitignore
│── images/
│   ├── class_dist.png
│   ├── feature_heatmap.png
│   ├── loss.png
│   ├── accuracy.png
│   └── conf_matrix.png
```

## How to Run

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook 2.-power-quality-fault-detection.ipynb
```

4. Run all cells to:
- load and preprocess the data,
- train the neural network,
- visualize learning curves,
- and evaluate classification performance.

## Highlights

- Electrical-engineering-focused ML project
- Feature-based power quality disturbance classification
- PyTorch implementation for tabular multiclass classification
- Useful EDA with class distribution and correlation heatmap
- Clear evaluation with loss, accuracy, and confusion matrix

## Future Improvements

- Compare the neural network against classical models such as Random Forest, XGBoost, or SVM
- Perform hyperparameter tuning for better generalization
- Add model explainability using SHAP or feature importance analysis
- Extend the workflow to real-time monitoring or deployment scenarios