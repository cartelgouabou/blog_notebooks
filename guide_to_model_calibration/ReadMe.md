# Enhancing Medical Predictions: A Comprehensive Guide to Model Calibration

## Introduction

This repository complements the Medium article titled ["Enhancing Medical Predictions: A Comprehensive Guide to Model Calibration"](https://medium.com/@cartelgouabou/enhancing-medical-predictions-a-comprehensive-guide-to-model-calibration-3ea741be88d7) which delves into the importance of model calibration in medical predictions and provides a detailed, step-by-step guide to implementing these techniques using Python. The primary focus of this script is to illustrate the concept of calibration rather than to optimize the model's performance.

## Purpose of the Script

The main goal of this script is to demonstrate how to assess and enhance the calibration of a predictive model. Using a publicly available Chest X-ray dataset from Kaggle, the script walks through the process of training a neural network model, evaluating its calibration, and applying calibration techniques such as Platt Scaling. The emphasis is on understanding and improving model calibration to ensure reliable and accurate predictions in medical applications.

## Getting Started

Follow these steps to set up the project environment and run the scripts:

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Download the Dataset
Download the Chest X-ray dataset from Kaggle. The dataset can be found at the following link:

[Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### 3. Install Requirements
Install the required Python packages:
```bash
pip install -r requirements.txt
```
### 4. Run the Notebook
Execute the Jupyter notebook to follow the implementation and results:
```bash
jupyter notebook chest_classification.ipynb
```

## Code Structure
The project consists of several key components, each serving a specific function in the process of model training and evaluation:

DataExplorer: Handles the exploration and visualization of dataset statistics.
ChestXRayClassifier: Manages data loading, model building, training, and calibration processes.
### Key Functions

- `__init__(self, img_size=224, batch_size=64)`: Initializes with image size and batch size.
- `load_data(self, train_dir, val_dir, test_dir)`: Loads and preprocesses data using ImageDataGenerator.
- `build_model(self, initial_layer_freezed_ratio=0.80, learning_rate=0.001)`: Builds the DenseNet121 model with custom layers on top.
- `train_model(self, epochs=10, early_stopping_patience=10)`: Trains the model with early stopping and learning rate reduction.
- `generate_predictions(self, subset='val')`: Generates predictions on validation or test set.
- `enhance_calibration(self)`: Applies Platt Scaling for calibration.
- `expected_calibration_error(self, samples, true_labels, M=5)`: Computes Expected Calibration Error (ECE).
- `evaluate_and_plot(self, preds, y_true, title, pos)`: Evaluates metrics and plots calibration curve.
- `combined_calibration_plots(self)`: Generates combined calibration plots before and after Platt Scaling.
- `save_model(self, root_model_name)`: Saves the model architecture and weights.

## Conclusion
This project serves as a practical guide to understanding and implementing model calibration in the context of medical predictions using deep learning. By following the steps outlined in this repository, users can gain insights into the calibration process and apply these techniques to their own predictive models.
