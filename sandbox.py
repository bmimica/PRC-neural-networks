import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import torch
    import torch.nn as nn

    from src.layers import multi_attention, AttentionBlock
    from src.architecture import Sequential
    from src.train import load_data, main

    import pandas as pd
    import pickle as pl
    from sklearn.model_selection import train_test_split

    import subprocess
    import os


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
    return label, model


@app.cell
def _(X_val, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch =  X_val[1 : 1 + 16].to(device)
    output = model(batch)
    output.shape
    return


@app.cell
def _(label, model):
    import sys

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


if __name__ == "__main__":
    app.run()
