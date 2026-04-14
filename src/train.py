import argparse

import torch
import torch.nn.functional as F

import pickle as pl
import pandas as pd
from sklearn.model_selection import train_test_split

import subprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(rand_seed, data_test_size = 0.2, data_eval_size = 0.1):
    y, data_df, pathway_gene, pathway, cancer_name = pl.load(open('pathway_data.pckl', 'rb'))
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

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(args.rand_seed)
    model = torch.load(args.model_path).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss_list = [] # tracks error for each epoch
    val_loss_list = []  
    accuracy_history = []
    mcc_history = []
    for epoch in range(args.epochs):
        model.train() # sets "training mode"
        train_loss = 0
        # shuffle data on each epoch to avoid bias
        permutation = torch.randperm(X_train.shape[0])

        """
        starts model train by batch
        """
        batch_size = args.batch_size
        for batch_idx, i in enumerate(range(0, X_train.shape[0], batch_size)):
            indices = permutation[i: i+batch_size]
            batch_x, batch_y = X_train[indices].to(device), Y_train[indices].to(device)

            # 
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = F.nll_loss(y_pred, batch_y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        """
        after finishing every batch do validation process: evaluates model performance on val data 
        """
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                batch_x = X_val[i : i + batch_size].to(device)
                batch_y = Y_val[i : i + batch_size].to(device)

                pred = model(batch_x)
                val_loss += F.nll_loss(pred, batch_y, reduction='sum').item()
                
                pred_choice = pred.argmax(dim=1, keepdim=True)
                correct += pred_choice.eq(batch_y.view_as(pred_choice)).sum().item()

        val_loss /= len(X_val)
        accuracy = 100. * correct / len(X_val)

        print(f"Run: {args.run_name} | Epoch {epoch}: Val Loss: {val_loss:.4f}, Acc: {accuracy:.2f}%")
    
    output_path = f"{args.run_name}_final.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Final model weights saved to {output_path}")

if __name__ == "__main__":
    main()