import os
import argparse
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime
from models import CrossAttnRNN, CrossAttnRNNZero
from dataset import Visuelle2


def run(args):
    print(args)

    # Seed for reproducibility (By default we use the number 21)
    pl.seed_everything(args.seed)

    ####################################### Load data #######################################
    # Load train-test data
    train_df = pd.read_csv(
        os.path.join(args.dataset_path, "stfore_train.csv"),
        parse_dates=["release_date"],
    )

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
    if demand:
        visuelle_pt_train = "visuelle2_train_processed_demand.pt"  
        visuelle_pt_test = "visuelle2_test_processed_demand.pt"  
    else:
        visuelle_pt_train = "visuelle2_train_processed_stfore.pt"
        visuelle_pt_test = "visuelle2_test_processed_stfore.pt"

    # Create (PyTorch) dataset objects
    trainset = Visuelle2(
        train_df,
        img_folder,
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len,
        demand,
        local_savepath=os.path.join(args.dataset_path, visuelle_pt_train)
    )
    testset = Visuelle2(
        test_df,
        img_folder,
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len,
        demand,
        local_savepath=os.path.join(args.dataset_path, visuelle_pt_test)
    )

    # # If you wish to debug with less data you can use this syntax
    trainset = torch.utils.data.Subset(trainset, list(range(1000)))
    testset = torch.utils.data.Subset(testset, list(range(1000)))

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(f"Train batches: {len(trainloader)}, test batches: {len(testloader)}")

    # ####################################### Train and eval model #######################################
    # Create model
    model = None
    if demand:
        model = CrossAttnRNNZero.CrossAttnRNN(
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
        )
    else:
        model = CrossAttnRNN.CrossAttnRNN(
            attention_dim=args.attention_dim,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            cat_dict=cat_dict, 
            col_dict=col_dict, 
            fab_dict=fab_dict,
            store_num=125, #This represents the maximum encoded value of the store id, the actuall nr of stores available in the dataset is 110, but this is needed for the nn.Embedding layer to work
            use_img=bool(args.use_img), 
            use_att=bool(args.use_att), 
            use_date=bool(args.use_date),
            use_trends=bool(args.use_trends),
            task_mode = int(args.task_mode),
            out_len=10
        )       

    # Define model saving procedure
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model_savename = args.model_type + "-" + args.wandb_run
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_dir + "/" + args.model_type,
        filename=model_savename + "---{epoch}---" + dt_string,
        monitor="val_mae",
        mode="min",
        save_top_k=1,
    )

    if args.use_wandb:
        wandb_logger = pl_loggers.WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run
        )

    trainer = pl.Trainer(
        gpus=[args.gpu_num],
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger if args.use_wandb else None,
        callbacks=[checkpoint_callback]
    )

    # Fit model
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)

    # Print out path of best model
    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='/media/data/gskenderi/visuelle2/')
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--new_product", type=int, default=0,
    help="Boolean variable to optionally use the dataset for the new product demand forecasting task (forecasting without a known past)")
    
     # Model specific arguments
    parser.add_argument("--model_type", type=str, default="RNN")
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--attention_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=9)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_att", type=int, default=0)
    parser.add_argument("--use_date", type=int, default=0)
    parser.add_argument("--use_trends", type=int, default=0)
    parser.add_argument("--task_mode", type=int, default=0, help="0-->2,1 - 1-->2,10")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gpu_num", type=int, default=0)

    # wandb arguments
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt/")

    
    args = parser.parse_args()
    run(args)
