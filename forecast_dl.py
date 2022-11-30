import os
import argparse
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from dataset import Visuelle2
from models.CrossAttnRNN21 import CrossAttnRNN as Model21
from models.CrossAttnRNN210 import CrossAttnRNN as Model210
from models.CrossAttnRNNDemand import CrossAttnRNN as DemandModel
from tqdm import tqdm
from utils import calc_error_metrics

def run(args):
    print(args)

    # Seed for reproducibility (By default we use the number 21)
    pl.seed_everything(args.seed)

    ####################################### Load data #######################################
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

    demand = bool(args.new_product)
    img_folder = os.path.join(args.dataset_path, 'images')

    visuelle_pt_test = "visuelle2_test_processed_demand.pt" if demand else "visuelle2_test_processed_stfore.pt"

    # Create (PyTorch) dataset objects
    testset = Visuelle2(
        test_df,
        img_folder,
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        52,
        demand,
        local_savepath=os.path.join(args.dataset_path, visuelle_pt_test)
    )

    # # If you wish to debug with less data you can use this syntax
    # trainset = torch.utils.data.Subset(trainset, list(range(1000)))
    # testset = torch.utils.data.Subset(testset, list(range(1000)))
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    print(f"Test batches: {len(testloader)}")

    # ####################################### Train and eval model #######################################
    # Load model
    model_savename = args.ckpt_path
    if demand:
        model = DemandModel(
            attention_dim=args.attention_dim,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_trends=args.num_trends,
            cat_dict=cat_dict, 
            col_dict=col_dict, 
            fab_dict=fab_dict,
            store_num=125, #This represents the maximum encoded value of the store id, the actuall nr of stores available in the dataset is 110, but this is needed for the nn.Embedding layer to work
            use_img=bool(args.use_img), 
            use_att=bool(args.use_att), 
            use_date=bool(args.use_date),
            use_trends=bool(args.use_trends),
            out_len=12
        ).load_from_checkpoint(model_savename)
    else:
        if args.task_mode == 0:
            model = Model21(
                attention_dim=args.attention_dim,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                use_img=args.use_img,
                out_len=args.output_len,
            )
        else:
            model = Model210(
                attention_dim=args.attention_dim,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                use_img=args.use_img,
                out_len=args.output_len,
                use_teacher_forcing=args.use_teacher_forcing,
                teacher_forcing_ratio=args.teacher_forcing_ratio,
            )

        
    model.load_state_dict(torch.load(model_savename)['state_dict'])
    model.to('cuda:'+str(args.gpu_num)) 
    model.eval()

    gt, forecasts = [], []
    for data in tqdm(testloader, total=len(testloader)):
        with torch.no_grad():
            (X, y, _, _, _, _, _, gtrends), images = data
            X, y, images = X.to("cuda:"+str(args.gpu_num)), y.to("cuda:"+str(args.gpu_num)), images.to("cuda:"+str(args.gpu_num))
            y_hat, _ = model(X, y, images)
            forecasts.append(y_hat)
            gt.append(y)

    norm_scalar = np.load(os.path.join(args.dataset_path, 'stfore_sales_norm_scalar.npy'))
    gt, forecasts = (
        torch.cat(gt).squeeze().detach().cpu().numpy() * norm_scalar,
        torch.cat(forecasts).squeeze().detach().cpu().numpy() * norm_scalar,
    )
    mae, wape = calc_error_metrics(gt, forecasts)
    print(f"{wape},{mae}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='visuelle2/')
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--new_product", type=int, default=0,
    help="Boolean variable to optionally use the dataset for the new product demand forecasting task (forecasting without a known past)")
    
    # Model specific arguments
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_len", type=int, default=10)
    parser.add_argument("--use_img", type=bool, default=True)
    parser.add_argument("--task_mode", type=int, default=0, help="0-->2-1 - 1-->2-10")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--use_teacher_forcing", action='store_true')
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.3)

    # wandb arguments
    parser.add_argument("--ckpt_path", type=str, default="ckpt/model.ckpt")
    
    args = parser.parse_args()
    run(args)
