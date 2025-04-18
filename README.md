# Limit Order Book Factor Extraction using Deep Learning

This repository implements deep learning models for extracting predictive features from Limit Order Book (LOB) data, with the goal of enhancing factor investing strategies. 

---

## Project Overview
We aim to replicate and extend findings from academic research by applying various deep learning architectures to LOB data. Our models predict short-term price movements by learning from historical order book snapshots.

Models implemented:
- **CNN** (Convolutional Neural Network)
- **LSTM** (Long Short-Term Memory Network with Attention)
- **MLP** (Multi-Layer Perceptron)
- **CNN-LSTM Hybrid**

Dataset used: [FI-2010 Benchmark Dataset]([https://github.com/zcaceres/deep-learning-limit-order-book](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649))

---

## Repository Structure

| File | Description |
|:----|:------------|
| `cnn.py` | CNN model for limit order book classification. |
| `lstm.py` | LSTM with attention mechanism for sequence modeling. |
| `mlp.py` | MLP baseline using flattened LOB inputs. |
| `cnn_lstm.py` | CNN + LSTM hybrid model inspired by DeepLOB. |
| `fi_loader.py` | Custom PyTorch Dataset class to load and preprocess FI-2010 data. |
| `utils.py` | Utility functions for saving and loading models. |
| `train.py` | Batch training loop with loss and accuracy tracking. |
| `train_model.ipynb` | Jupyter notebook to train models interactively. |
| `evaluate_models.ipynb` | Notebook to evaluate saved models on test data. |
