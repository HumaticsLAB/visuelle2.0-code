import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import TensorDataset
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Visuelle2:
    def __init__(
        self,
        sales_df,
        img_root,
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len,
        demand,
        local_savepath
    ):
        self.sales_df = sales_df
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_root = img_root
        self.demand = demand

        print("Loading dataset...")
        if os.path.isfile(local_savepath):
            self.dataset = torch.load(local_savepath) # load dataset directly from saved files
        else:
            self.dataset = self.preprocess_data() # If file doesn't exist or you wish to re-process/create from scratch 
            torch.save(self.dataset, local_savepath)
        print("Done.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Define image transformations
        img_transforms = Compose(
            [
                Resize((299, 299)), # Inception v3 uses 299x299 images: https://arxiv.org/abs/1512.00567
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Read image (online with each getitem call, in this way we spare memory)
        img_path = self.sales_df.loc[idx, "image_path"]
        img = Image.open(os.path.join(self.img_root, img_path)).convert("RGB") # Make sure to use RGB
        pt_img = img_transforms(img)

        return self.dataset[idx], pt_img

    def frame_series(self, train_window=2, forecast_horizon=1):
        X, y = [], []
        sales_and_restocks = self.sales_df.copy(deep=True).iloc[:, -13:]
        restocks, sales = (
            sales_and_restocks.values[:, 0],
            sales_and_restocks.values[:, -12:],
        )

        # Check for cases where the sold amount goes over the restocking total
        # This part corresponds to the selection of the legit sale signals
        clean_ts, split_idx = [], []
        for i, ts in enumerate(sales):
            item_stock = restocks[i]
            if ts.sum() <= item_stock:  # If it's under ok
                clean_ts.append(ts)
                split_idx.append(0)
            else:
                # Check when the sales go over the total and drop those values
                sidx = np.where(ts.cumsum() > item_stock)[0][0]
                ts[ts.cumsum() > item_stock] = 0
                clean_ts.append(ts)
                split_idx.append(sidx)

        clean_ts, split_idx = np.array(clean_ts), torch.tensor(split_idx)
        self.split_idx = split_idx

        # Frame the time series (based on the specified windows)
        for x in tqdm(clean_ts, total=clean_ts.shape[0], desc="Framing time series"):
            features, target = [], []
            for i in range(x.shape[0] - train_window - forecast_horizon + 1):
                features.append(torch.tensor(x[i : i + train_window]))
                target.append(
                    torch.tensor(
                        x[i + train_window : i + train_window + forecast_horizon]
                    )
                )

            features_tensor, target_tensor = torch.stack(features), torch.stack(target)
            X.append(features_tensor), y.append(target_tensor)

        return torch.stack(X).type(torch.float32), torch.stack(y).type(torch.float32)

    def preprocess_data(self):
        if self.demand:
            ts = self.sales_df.copy(deep=True).iloc[:, -12:].values
            ts = torch.tensor(ts).type(torch.float32)
        else:
            X, y = self.frame_series()
        
        # Get the Gtrends time series associated with each product
        # Read the images (extracted image features) as well
        print("Loading exogenous time series...")
        gtrends = []
        for (_, row) in tqdm(self.sales_df.iterrows(), total=len(self.sales_df)):
            cat, col, fab, start_date = (
                row.category,
                row.color,
                row.fabric,
                row.release_date,
            )

            # Get the gtrend signal up to the previous year (52 weeks) of the release date
            gtrend_start = start_date - pd.DateOffset(weeks=52)
            cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[
                : self.trend_len
            ]
            col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[
                : self.trend_len
            ]
            fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[
                : self.trend_len
            ]

            # Edge cases with missing trends
            if len(cat_gtrend) < self.trend_len:
                cat_gtrend = self.gtrends.loc[:start_date][cat][-52:].values[
                    : self.trend_len
                ]
            if len(col_gtrend) < self.trend_len:
                col_gtrend = self.gtrends.loc[:start_date][col][-52:].values[
                    : self.trend_len
                ]
            if len(fab_gtrend) < self.trend_len:
                fab_gtrend = self.gtrends.loc[:start_date][fab][-52:].values[
                    : self.trend_len
                ]

            # Normalize trends locally for each case, they have a relative magnitude in a particular timespan
            cat_gtrend = (
                MinMaxScaler().fit_transform(cat_gtrend.reshape(-1, 1)).flatten()
            )
            col_gtrend = (
                MinMaxScaler().fit_transform(col_gtrend.reshape(-1, 1)).flatten()
            )
            fab_gtrend = (
                MinMaxScaler().fit_transform(fab_gtrend.reshape(-1, 1)).flatten()
            )

            # Create multivariate series
            multitrends = np.vstack([cat_gtrend, col_gtrend, fab_gtrend])
            gtrends.append(multitrends)

        # Extract temporal features
        release_date = self.sales_df.loc[:, "release_date"]
        day, week, month, year = (
            torch.LongTensor(release_date.dt.day.values),
            torch.LongTensor(release_date.dt.isocalendar().week.values),
            torch.LongTensor(release_date.dt.month.values),
            torch.LongTensor(release_date.dt.year.values),
        )
        temporal_features = torch.vstack([day, week, month, year]).T
        temporal_features = temporal_features / temporal_features.max(0)[0] # Scale temporal features by respective maximum val

        categories, colors, fabrics, stores = (
            torch.LongTensor(
                [self.cat_dict[val] for val in self.sales_df.loc[:, "category"].values]
            ),
            torch.LongTensor(
                [self.col_dict[val] for val in self.sales_df.loc[:, "color"].values]
            ),
            torch.LongTensor(
                [self.fab_dict[val] for val in self.sales_df.loc[:, "fabric"].values]
            ),
            torch.LongTensor(
                self.sales_df.loc[:, "retail"].values
            ),
        )

        gtrends = torch.FloatTensor(gtrends).type(torch.float32)

        forecasting_dataset = None
        if self.demand:
            forecasting_dataset = TensorDataset(ts, categories, colors, fabrics, stores, temporal_features, gtrends)
        else:
            forecasting_dataset = TensorDataset(X, y, categories, colors, fabrics, stores, temporal_features, gtrends)

        return forecasting_dataset

