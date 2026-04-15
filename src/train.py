import argparse

import torch
import torch.nn.functional as F
import numpy as np

import pickle as pl
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt

import subprocess
import os
import sys

if 'marimo' in sys.modules or 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(rand_seed, data_test_size = 0.2, data_eval_size = 0.1):
    with open('pathway_data.pckl', 'rb') as f:
        y, data_df, pathway_gene, pathway, cancer_name = pl.load(f)
    gene_list = data_df.columns.tolist()
    data_df['cancer_type'], uniques = pd.factorize(y)

    # keeps the 20% ratio for every cancer type automatically.
    train_df_, test_df = train_test_split(
        data_df, 
        test_size = data_test_size, 
        stratify = data_df['cancer_type'], 
        random_state = rand_seed
    )

    train_df, val_df = train_test_split(
        train_df_, 
        test_size = data_eval_size, # takes 10% of of the training set for validation
        stratify = train_df_['cancer_type'], 
        random_state=42
    )

    # torch tensors 
    X_train = torch.tensor(train_df[gene_list].values, dtype=torch.float32)
    Y_train = torch.tensor(train_df['cancer_type'].values, dtype=torch.long)

    X_val = torch.tensor(val_df[gene_list].values, dtype=torch.float32)
    Y_val = torch.tensor(val_df['cancer_type'].values, dtype=torch.long)

    X_test = torch.tensor(test_df[gene_list].values, dtype=torch.float32)
    Y_test = torch.tensor(test_df['cancer_type'].values, dtype=torch.long)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def train(model, batch_size = 16, lr=0.001, epochs=50, label="exp_1"):
    model_path = f"temp_{label}.pth"
    torch.save(model, model_path)

    cmd = [
        "python", "train.py",
        "--model_path", model_path,
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--run_name", label
    ]

    # run in terminal
    subprocess.Popen(cmd)
    print(f"training model: {label}. Results will save to {label}.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required = True) # reads the model from a saved .pth file
    parser.add_argument("--lr", type=float, default=0.001) # learning rate
    parser.add_argument("--rand_seed", default=52, type=int, help="random seed used to split train test and val ",)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run_name", type=str, required = True) # label of a certain training process
    args = parser.parse_args()

    results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(results_dir):
          os.makedirs(results_dir)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(args.rand_seed)
    model = torch.load(args.model_path, weights_only = False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss_list = [] # tracks error for each epoch
    val_loss_list = []  
    acc_list = []
    auc_list = []
    mcc_list = []
    f1_list = []
    confusion_matrix_list = []

    print("--- TRAINING ---")
    epoch_pbar = tqdm(range(args.epochs), desc="Epochs", position = 0)
    for epoch in epoch_pbar:
        model.train() # sets "training mode"
        train_loss = 0
        # shuffle data on each epoch to avoid bias
        permutation = torch.randperm(X_train.shape[0])

        """
        starts model train by batch
        """
        batch_size = args.batch_size

        batch_pbar = tqdm(enumerate(range(0, X_train.shape[0], batch_size)), 
                        total=X_train.shape[0]//batch_size, 
                        desc=f"Epoch {epoch+1}", 
                        position = 1)
        for batch_idx, i in batch_pbar:
            indices = permutation[i: i+batch_size]
            batch_x, batch_y = X_train[indices].to(device), Y_train[indices].to(device)

            # 
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = F.nll_loss(y_pred, batch_y, reduction = 'sum') 

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(X_train)
        train_loss_list.append(train_loss)

        """
        after finishing every batch do validation process: evaluates model performance on validation data 
        """
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            y_val=[] # to save y values temporarily
            out_val = [] # to save predicted y values and compare
            prob_out_val = [] # to see the actual "probability" of a certain label result

            for i in range(0, X_val.shape[0], batch_size):
                batch_x = X_val[i : i + batch_size].to(device)
                batch_y = Y_val[i : i + batch_size].to(device)

                pred = model(batch_x.float()) # predictions of the input x
                val_loss += F.nll_loss(pred, batch_y, reduction='sum').item() # calculates the loss
                
                pred_choice = pred.argmax(dim=1, keepdim=True)
                correct += pred_choice.eq(batch_y.view_as(pred_choice)).sum().item()

                y_val.append(batch_y.cpu().numpy())
                out_val.append(pred_choice.cpu().numpy())
                prob_out_val.append(torch.exp(pred).cpu().numpy())


            val_loss /= len(X_val)
            val_loss_list.append(val_loss)
            accuracy = 100. * correct / len(X_val)

            # results that i want to export:  confusion matrix, 
            y_val = np.concatenate(y_val).ravel()
            out_val = np.concatenate(out_val).ravel()
            prob_out_val = np.vstack(prob_out_val)

            acc_val = metrics.accuracy_score(y_val, out_val)
            f1 = metrics.f1_score(y_val, out_val, average='micro')
            confusion_mat = metrics.confusion_matrix(y_val, out_val)
            mcc = metrics.matthews_corrcoef(y_val, out_val)

            encoder_ = LabelBinarizer()
            y_val = encoder_.fit_transform(y_val)
            roc_auc = metrics.roc_auc_score(y_val, prob_out_val, multi_class='ovr', average='micro')

        """
        save data of interest
        """
        confusion_matrix_list.append(confusion_mat)
        mcc_list.append(mcc)
        acc_list.append(acc_val) # accuracy score
        auc_list.append(roc_auc) #
        f1_list.append(f1)

        print(f"Run: {args.run_name} | Epoch {epoch}: Val Loss: {val_loss:.4f}, Acc: {accuracy:.2f}%")
    
    output_path = f"{args.run_name}_final.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Final model weights saved to {output_path}")

    run_history = {
        "train_loss": train_loss_list, # train loss vs n epochs
        "val_loss": val_loss_list,  # val loss vs n epochs
        "accuracy": acc_list, 
        "f1": f1_list,
        "mcc": mcc_list,
        "auc": auc_list,
        "confusion_matrix": confusion_matrix_list
    }

    # Using args.run_name ensures you don't overwrite previous tests
    res_file = os.path.join(results_dir, f"res_{args.run_name}.pkl")
    fig_file = os.path.join(results_dir, f"fig_{args.run_name}.png")
    with open(res_file, 'wb') as f:
        pl.dump(run_history, f)

    # 3. Quick PNG for a "Sanity Check"
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Train')
    plt.plot(val_loss_list, label='Val')
    plt.title(f'Loss: {args.run_name}')
    plt.legend()
    plt.savefig(fig_file)
    plt.close()
    
    print(f"Done! Results saved to {res_file}. Check quick_check_{args.run_name}.png to see if it learned.")

if __name__ == "__main__":
    main()