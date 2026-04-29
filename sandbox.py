import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import torch
    import torch.nn as nn

    from src.layers import multi_attention, AttentionBlock, echo_state
    from src.architecture import Sequential, LeakyResidualConnector
    from src.train import load_data, main

    import pandas as pd
    import pickle as pl
    from sklearn.model_selection import train_test_split

    import subprocess
    import os

    import sys

    import pandas as pd
    from pathlib import Path

    import matplotlib.pyplot as plt

    from sklearn.metrics import ConfusionMatrixDisplay

    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data pre-processing
    """)
    return


@app.cell
def _():
    rand_seed_0 = 42
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(rand_seed = rand_seed_0)
    return X_test, X_val


@app.cell
def _(X_test):
    X_test[0].shape
    return


@app.cell
def _():
    tensor = torch.eye(4)
    tensor.unsqueeze(0).unsqueeze(0).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Neural networks
    """)
    return


@app.cell
def _():
    # parameters
    batch_size = 16
    n_head = 5
    n_gene = 1708
    n_feature = n_gene 
    mode = 0
    n_class = 34


    #architecture
    layers = [
        AttentionBlock(batch_size, n_head, n_gene, n_feature, mode),
        AttentionBlock(batch_size, n_head, n_gene, n_feature, mode),
        AttentionBlock(batch_size, n_head, n_gene, n_feature, mode),
        nn.ReLU(),
        nn.Linear(n_gene, n_class),
        nn.LogSoftmax(dim = 1)
    ]

    label = 'test_2'
    model  = Sequential(layers, label)
    model.save()
    return batch_size, label, model, n_class, n_head


@app.cell
def _(X_val, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch =  X_val[1 : 1 + 16].to(device)
    output = model(batch)
    output.shape
    return


@app.cell(disabled=True)
def _(label, model):
    # We manually set the "command line arguments" for this cell only
    sys.argv = [
        "train.py", # the train script to be run 
        "--model_path", model.path, 
        "--run_name", f"{label}", # label of the results 
        "--epochs", "10", 
        "--batch_size", "16", 
        "--lr", "0.0001"
    ]

    main()
    return


@app.cell
def _(batch_size, n_class, n_head):
    # batch_size = 16
    # n_head = 5
    fan_in =  1708
    fan_out = 1708
    R_size  = 256
    # n_class = 34
    # mode = 0

    #architecture
    layers_2 = [
        echo_state(batch_size, n_head, fan_in, fan_out, R_size),
        nn.ReLU(),
        nn.Linear(fan_in, n_class),
        nn.LogSoftmax(dim = 1)
    ]

    label_2 = 'test_2_esn'
    model_2  = Sequential(layers_2, label_2)
    model_2.save()
    return label_2, model_2


@app.cell
def _(X_test, model_2):
    model_2(X_test)
    return


@app.cell
def _(label_2, model_2):
    sys.argv = [
        "train.py", # the train script to be run 
        "--model_path", model_2.path, 
        "--run_name", f"{label_2}", # label of the results 
        "--epochs", "25", 
        "--batch_size", "16", 
        "--lr", "0.0001"
    ]

    main()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data Analysis
    """)
    return


@app.cell
def _():
    results_dir =  Path(r"F:\benja\project_TGEM\reproduccion resultados\results")

    for file_path in results_dir.iterdir():
            if file_path.suffix == ".pkl":
                print(file_path.name)
    return (results_dir,)


@app.cell
def _(results_dir):
    results_esn = pd.read_pickle(results_dir / 'res_test_2_esn.pkl')
    results_attention = pd.read_pickle(results_dir / 'res_test_2.pkl')


    results = [results_esn, results_attention]
    results_esn.keys()
    return (results,)


@app.cell
def _(results):
    labels = ['esn', 'attention']
    colors = ['red', 'blue']
    key =  'f1'

    for i, result in enumerate(results):
        plt.plot(result[key], label = labels[i], color = colors[i])

    plt.title(key) 
    plt.legend()
    plt.show()
    return


@app.cell
def _(results):
    def _():
        epoch = 10

        for i, result in enumerate(results):
            cm = np.array(result['confusion_matrix'][epoch-1])
            fig, ax = plt.subplots(figsize=(10, 10))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
            disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
        
            plt.xticks(rotation=90)
            plt.title(f"Confusion Matrix - Epoch {epoch}")
        return plt.show()


    _()
    return


if __name__ == "__main__":
    app.run()
