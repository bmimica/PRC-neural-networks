import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import torch
    import torch.nn as nn

    from layers import multi_attention, AttentionBlock
    from architecture import Sequential
    from train import load_data

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
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()
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

    model  = Sequential(layers)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
