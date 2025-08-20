
# Unsupervised Anomaly Detection in Time-Series Data using a Variational Autoencoder (VAE)

This project demonstrates an end-to-end workflow for unsupervised anomaly detection in multivariate time-series data using a Variational Autoencoder (VAE) implemented in PyTorch. The script simulates sensor data from a power converter, trains a VAE exclusively on normal operational data, and then uses multiple scoring techniques to identify anomalous behavior in a test dataset.

## Table of Contents

  - [Overview](https://www.google.com/search?q=%23overview)
  - [Project Workflow](https://www.google.com/search?q=%23project-workflow)
  - [Core Concepts: How Anomaly Detection Works](https://www.google.com/search?q=%23core-concepts-how-anomaly-detection-works)
  - [Getting Started](https://www.google.com/search?q=%23getting-started)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation](https://www.google.com/search?q=%23installation)
  - [How to Run](https://www.google.com/search?q=%23how-to-run)
  - [Code Structure](https://www.google.com/search?q=%23code-structure)
  - [Key Outputs and Visualizations](https://www.google.com/search?q=%23key-outputs-and-visualizations)
  - [Customization](https://www.google.com/search?q=%23customization)

## Overview

In industrial settings like power electronics, identifying system failures or deviations from normal behavior is critical for maintenance and safety. This project tackles this problem using an unsupervised deep learning approach. Since anomalous data is often rare and diverse, the model is trained only on normal data. The core idea is that a model trained to reconstruct "normal" signals will perform poorly when trying to reconstruct an "abnormal" one, and this discrepancy can be used as an anomaly score.

### Key Features

  - **Data Simulation:** Generates realistic, multivariate time-series data for a power converter, including normal operation and three distinct types of anomalies (voltage spikes, current oscillations, and temperature drift).
  - **VAE Implementation:** A robust Variational Autoencoder built with PyTorch, designed to learn a compressed, latent representation of the input data.
  - **Multiple Anomaly Metrics:** Implements and compares three different anomaly scoring methods:
    1.  **Reconstruction Error (MSE):** The primary measure of how well the VAE can reproduce the input.
    2.  **Latent Space Euclidean Distance:** Measures how far a data point's latent representation is from the "center" of normal data.
    3.  **Latent Space Mahalanobis Distance:** A more sophisticated distance metric that accounts for the variance and covariance of the latent space, making it sensitive to deviations along less variant axes.
  - **Comprehensive Evaluation:** Uses ROC curves, Precision-Recall (PR) curves, classification reports, and confusion matrices to thoroughly evaluate the detection performance of each scoring method.
  - **Rich Visualizations:** Produces a suite of plots to help understand the data, model performance, and results, including latent space distribution and sample reconstruction quality.

## Project Workflow

The script follows a clear, sequential workflow from data generation to final evaluation:

1.  **Data Generation:** Normal and anomalous time-series data for voltage, current, and temperature sensors are simulated.
2.  **Data Preprocessing:** The data is flattened, split into training and testing sets, and scaled to a `[0, 1]` range. Crucially, the VAE is trained *only on normal data*.
3.  **Model Training:** The VAE model is trained to minimize a combined loss function, which includes both reconstruction error and the Kullback-Leibler (KL) divergence. The loss is defined as:
    ## L(θ,ϕ)=Eqϕ(z∣x)[logpθ(x∣z)]-DKL(qϕ(z∣x)∣∣p(z))
4.  **Anomaly Score Calculation:** The trained VAE is used to process the test set (containing both normal and anomalous data) to calculate the three distinct anomaly scores.
5.  **Performance Evaluation:** The effectiveness of each score in distinguishing normal from anomalous data is measured using Area Under the Curve (AUC) for both ROC and PR curves.
6.  **Thresholding and Classification:** A practical threshold is determined (e.g., the 99th percentile of the reconstruction error on normal training data) to classify test samples and generate a final classification report and confusion matrix.

## Core Concepts: How Anomaly Detection Works

The VAE learns a compressed representation (the latent space) of the normal training data. The anomaly detection logic is based on two key principles:

1.  **Reconstruction Failure:** Because the VAE has only learned the patterns and distributions of normal data, it will struggle to accurately reconstruct a sample that deviates significantly from this norm. Therefore, anomalous samples will have a high reconstruction error.

2.  **Latent Space Distribution:** The encoder part of the VAE maps normal data to a tight, well-defined cluster in the latent space (approximating a Gaussian distribution). Anomalous samples, being unfamiliar to the encoder, are likely to be mapped to points far away from this central cluster. The Euclidean and Mahalanobis distances quantify this "farness."

## Getting Started

### Prerequisites

  - Python 3.7+
  - A modern environment that can run Jupyter notebooks or Python scripts (e.g., Google Colab, VS Code with Python extension, or a local Jupyter installation).

### Installation

The script includes a pip command to install all necessary libraries. You can also run this in your terminal:

```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib seaborn scipy
```

## How to Run

The script is structured as a Colab/Jupyter notebook, with code divided into executable cells.

1.  Upload or open the `encoder.py` file in a compatible environment (like Google Colab).
2.  Execute each cell in sequence from top to bottom.
3.  The script will print status updates to the console and generate plots inline.

## Code Structure

The script is organized into logical cells, each performing a specific task:

  - **Cell 1: Setup and Imports:** Installs dependencies and imports all necessary libraries. Sets up the device (GPU/CPU) and random seeds for reproducibility.
  - **Cell 2: Data Generation:** Simulates the power converter sensor data, including normal and anomalous samples. It also handles data splitting, scaling, and conversion to PyTorch `DataLoader` objects.
  - **Cell 3: VAE Model Definition:** Contains the `VAE` class, which defines the encoder, decoder, and reparameterization logic using `torch.nn.Module`.
  - **Cell 4: VAE Loss and Training:** Defines the VAE loss function (MSE + KLD) and contains the main training loop that optimizes the model.
  - **Cell 5: Anomaly Score Calculation:** After training, this cell passes the test data through the model to compute the reconstruction errors and latent space distances.
  - **Cell 6: Performance Evaluation:** Calculates the ROC AUC and PR AUC for each of the three anomaly scores.
  - **Cell 7: Visualization of Performance:** Generates the ROC and PR curve comparison plots. If the latent dimension is 2, it also plots a scatter visualization of the latent space.
  - **Cell 8: Advanced Analysis:** Visualizes how well the VAE reconstructs different types of samples (one normal, three anomalous) and performs a practical thresholding analysis to generate a final classification report and confusion matrix.

## Key Outputs and Visualizations

Upon running the script, you should expect the following outputs:

  - **Console Output:**

      - Confirmation of the device being used (CPU or CUDA).
      - Shapes of the generated and processed datasets.
      - Training loss for each epoch.
      - Calculated ROC AUC and PR AUC values for each anomaly score.
      - A classification report with precision, recall, and F1-score.

  - **Generated Plots:**

    1.  **Simulated Data Samples:** Example plots of normal vs. anomalous signals for each sensor.
    2.  **VAE Training Loss:** A line plot showing the decrease in loss over epochs.
    3.  **ROC and PR Curve Comparison:** Side-by-side plots comparing the performance of the three anomaly scores.
    4.  **Latent Space Visualization:** A 2D scatter plot (if `LATENT_DIM=2`) showing the separation of normal and anomalous data.
    5.  **Reconstruction Quality:** A 2x2 grid of plots showing the original vs. reconstructed signals for a normal sample and three types of anomalous samples.
    6.  **Confusion Matrix:** A heatmap visualizing the performance of the chosen threshold for classification.

## Customization

Several parameters at the top of the cells can be easily modified to experiment with the model and data:

  - **Data Simulation (Cell 2):**
      - `NUM_SAMPLES`, `TIME_STEPS`, `NUM_FEATURES`: Change the size and shape of the dataset.
      - Simulation parameters (e.g., `voltage_base`, `spike_start`) can be altered to create different types of signals.
  - **Model Architecture (Cell 3):**
      - `LATENT_DIM`: A crucial hyperparameter. Try `2` to get the 2D latent space visualization, or increase it for potentially better performance on complex data.
      - The number of neurons in the `nn.Linear` layers can be adjusted to change model capacity.
  - **Training (Cell 4):**
      - `LEARNING_RATE`, `NUM_EPOCHS`: Tune these to control the training process and prevent over/underfitting.
  - **Thresholding (Cell 8):**
      - `percentile`: Adjust the percentile (e.g., from `99` to `95`) to see how the trade-off between precision and recall changes.
