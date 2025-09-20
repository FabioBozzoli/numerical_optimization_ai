# Optimizer Comparison: SVRG vs. Adam vs. Adagrad

This project implements and compares three different optimization algorithms for neural network training: **Adam**, **Adagrad**, and a custom implementation of **Stochastic Variance Reduced Gradient (SVRG)**.

The analysis is performed on a binary classification task (cats vs. dogs) using a simple two-layer neural network with PyTorch. The script systematically explores different combinations of hyperparameters (learning rate and batch size) and generates comparative performance plots.


*(Note: The included plot image is an example. Running the script will generate your own updated results.)*

## 📜 Contents
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Hugging Face Authentication](#hugging-face-authentication)
- [Usage](#-usage)
- [Results](#-results)
- [Code Structure](#-code-structure)

## ✨ Features
- **Custom SVRG Implementation**: Includes a dedicated training function for SVRG, reducing gradient variance.
- **Comprehensive Benchmarking**: Trains and evaluates the model using both torch-native optimizers (Adam, Adagrad) and SVRG.
- **Grid Search**: Automatically tests different learning rates and batch sizes.
- **Result Visualization**: Generates three comparative plots with `matplotlib`:
    1.  Accuracy vs. Epochs
    2.  Accuracy vs. Training Time
    3.  Hinge Loss vs. Epochs
- **Streamlined Data Loading**: Uses `datasets` from Hugging Face for effortless dataset and split downloading.

## 📁 Dataset
The project utilizes a reduced subset of the **[microsoft/cats_vs_dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs)** dataset from the Hugging Face Hub:
- **Training Set**: 5,000 images.
- **Test Set**: 500 images.

All images are resized to 128x128 pixels and normalized. Labels are mapped to `-1` (cat) and `1` (dog) for Hinge Loss compatibility.

## 🤖 Model Architecture
The model is a simple two-layer fully connected neural network (`TwoLayerNet`):
1.  **Input Layer**: Accepts a flattened image (128 × 128 × 3 = 49152 features).
2.  **Hidden Layer**: Dense layer (500 units) with ReLU activation.
3.  **Output Layer**: Single neuron with a scalar output for binary classification.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    ```

2.  **(Recommended) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision matplotlib numpy datasets huggingface_hub scikit-learn
    ```
    *(Note: `scikit-learn` is used by `datasets`, and the `seaborn-v0_8-darkgrid` option is included in `matplotlib`)*

### Hugging Face Authentication
To download the dataset you must authenticate with Hugging Face.

⚠️ **IMPORTANT**: **Never insert your token directly into your code!**

Instead, run this command in your terminal and enter your token when prompted. It will be stored safely on your machine.

```bash
huggingface-cli login
```
You can find your token [in your Hugging Face account settings](https://huggingface.co/settings/tokens).

## ⚡ Usage
Once you’ve set up your environment and authentication, simply run the Python script:

```bash
python your_script_name.py
```

Progress and metrics for each optimizer and hyperparameter combination will be printed in your terminal. After completing, three comparison plots will open for interactive analysis.

## 📊 Results
At the end of execution, the script generates a three-panel figure for analyzing optimizer performance at a glance:
- **Convergence Speed**: Which optimizer reaches high accuracy with fewer epochs?
- **Computational Efficiency**: Which optimizer is faster in training time?
- **Loss Stability**: How does the loss evolve for each training method?

## 🧠 Code Structure
- `get_cats_vs_dogs_data()`: Loads, transforms, and prepares the dataset.
- `TwoLayerNet`: Definition of the network architecture (PyTorch).
- `train_native_optimizer()`: Training function for PyTorch-native optimizers (Adam, Adagrad) using **Hinge Loss**.
- `train_svrg()`: Training function implementing the **SVRG** algorithm.
- **Main section**:
    - Defines hyperparameters to scan (learning rates, batch sizes).
    - Loops over all combinations for thorough benchmarking.
    - Runs training and collects results.
    - Uses `matplotlib` to plot the final comparative figures.

---

**Note:**  
If you use this code for your own research or in production, please [remove your Hugging Face authentication token from the code](https://huggingface.co/docs/huggingface_hub/how-to-use-cli#login) and follow best practices for secrets management!
