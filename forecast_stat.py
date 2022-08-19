import os
import argparse
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from models.Oracle import Oracle
from dataset import Visuelle2
from utils import calc_error_metrics


def run(args):
    print(args)
    
    # Seed for reproducibility (By default we use the number 21)
    pl.seed_everything(args.seed)

    ####################################### Load data #######################################
    # Load train-test data
    test_df = pd.read_csv(
        os.path.join(args.dataset_path, "stfore_test.csv"),
        parse_dates=["release_date"],
    )

    # Load attribute encodings
    cat_dict = torch.load(os.path.join(args.dataset_path, "category_labels.pt"))
    col_dict = torch.load(os.path.join(args.dataset_path, "color_labels.pt"))
    fab_dict = torch.load(os.path.join(args.dataset_path, "fabric_labels.pt"))

    # Load Google trends
    gtrends = pd.read_csv(
        os.path.join(args.dataset_path, "vis2_gtrends_data.csv"), index_col=[0], parse_dates=True
    )

    img_folder = os.path.join(args.dataset_path, 'images')
    visuelle_pt_test = "visuelle2_test_processed_stfore.pt"

    # Create (PyTorch) dataset objects
    testset = Visuelle2(
        test_df,
        img_folder,
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len,
        demand=False, # Can't do demand forecasting of new products with these methods,
        local_savepath=os.path.join(args.dataset_path, visuelle_pt_test)
    )

    # # If you wish to debug with less data you can use this syntax
    # testset = torch.utils.data.Subset(testset, list(range(1000)))

    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    print("Test batches:", len(testloader))

    # ####################################### Run model #######################################
    # Create model
    model = Oracle(args.method, bool(args.use_teacher_forcing))

    # Perform forecasts
    gt, forecasts = [], []
    for data in tqdm(testloader, total=len(testloader)):
        with torch.no_grad():
            (X, y, _, _, _, _, _, _), _ = data
            y_hat = model(X)
            forecasts.append(y_hat)
            gt.append(y)

    norm_scalar = np.load(os.path.join(args.dataset_path, 'stfore_sales_norm_scalar.npy'))
    gt, forecasts = (
        torch.cat(gt).squeeze().numpy() * norm_scalar,
        torch.cat(forecasts).squeeze().numpy() * norm_scalar,
    )
    mae, wape = calc_error_metrics(gt, forecasts)
    print(f"Results for {args.method}")
    print(f"{wape},{mae}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--dataset_path", type=str, default='/media/data/gskenderi/visuelle2/')
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--method", type=str, default="naive")
    parser.add_argument("--use_teacher_forcing", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    run(args)
