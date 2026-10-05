# MLP from Scratch

A multilayer perceptron implemented from scratch with Python and NumPy for binary classification. The notebook follows the implementation from a single hidden layer to a network with two hidden layers, with manual gradient derivations and experiments on training behavior. Forward propagation, backpropagation, parameter updates, and evaluation metrics are implemented directly with NumPy. Matplotlib is used for plots.

**[Open the notebook](mlp-from-scratch.ipynb)** to explore the code, saved results, and analysis. The notebook is written in Portuguese, and this README provides an overview in English. Its outputs are saved, so you can read the results without running the experiments.

## Implementation

- An 80/20 train–validation split with a fixed seed, and normalization using training statistics only.
- ReLU and Swish hidden-layer activations, with a sigmoid output and binary cross-entropy (BCE).
- He initialization and comparisons with small-amplitude random initialization.
- Manual backpropagation for networks with one or two hidden layers.
- Full-batch gradient descent and SGD with different mini-batch sizes.
- Early stopping with restoration of the parameters giving the lowest validation loss.
- Accuracy, per-class precision, recall, F1-score, and confusion matrices.

## Experiments

The notebook investigates hidden-layer width, learning rate, activation function, weight initialization, batch size, and network depth. Repeated training runs with different seeds help assess how results vary with initialization and, for mini-batch training, sample order.

The final architecture is `2 → 64 → 64 → 1`, with ReLU in both hidden layers and a sigmoid output. It uses He initialization, mini-batches of 16 samples, a learning rate of `0.05`, and early stopping with a patience of 200 epochs.

## Validation results

The following results are recorded in the notebook's final architecture comparison. All configurations use mini-batch SGD with a batch size of 16 and the same 10 seeds. Values are **mean +- sample standard deviation**.

| Architecture | Accuracy | Macro F1-score | Validation BCE |
| --- | --- | --- | --- |
| `2 → 256 → 1` | 0.920 +- 0.017 | 0.919 +- 0.018 | 0.223 +- 0.014 |
| `2 → 16 → 16 → 1` | 0.924 +- 0.023 | 0.924 +- 0.023 | 0.226 +- 0.055 |
| `2 → 32 → 32 → 1` | 0.948 +- 0.017 | 0.947 +- 0.017 | 0.218 +- 0.034 |
| **`2 → 64 → 64 → 1`** | **0.948 +- 0.018** | **0.947 +- 0.018** | **0.210 +- 0.028** |
| `2 → 128 → 128 → 1` | 0.950 +- 0.014 | 0.950 +- 0.014 | 0.221 +- 0.022 |
| `2 → 256 → 256 → 1` | 0.949 +- 0.007 | 0.949 +- 0.007 | 0.225 +- 0.021 |

The network with two hidden layers of 64 units was selected for its lowest mean validation BCE. The 128-unit configuration had slightly higher accuracy, but increasing width beyond 64 units did not improve the selection criterion.

## Running the notebook

The notebook was developed with Python 3.12.3. It requires NumPy, Matplotlib, and a Jupyter environment. The JSON and timing utilities use Python's standard library.

From the repository directory, create an environment and install the dependencies (Linux/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib jupyterlab
jupyter lab mlp-from-scratch.ipynb
```

Keep `dataset_2.json` in the same directory as the notebook. To reproduce the experiments, select the environment's kernel and run the cells in order from a fresh kernel. Some functions are redefined as the architecture changes. Running all experiments includes multiple training runs, so total runtime depends on your machine.

## Dataset and academic context

The included [dataset_2.json](dataset_2.json) contains 600 samples, two features, and binary labels. The notebook uses 480 samples for training and 120 for validation, with a random split that is not explicitly stratified.

This project was developed as assignment 1 for **UFSM00993 - Deep Learning** at the Federal University of Santa Maria (UFSM). The dataset was provided by the professor. Its original source is [dataset_2.json in the course repository](https://github.com/tltsilveira/ufsm00993/blob/main/dataset_2.json).