import torch
import numpy as np
import pytorch_lightning as pl
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.forecasting.theta import ThetaModel

# This model uses time-series forecasting algorithms that rely only on the series itself
# As such it can be used firectly on the testing set and be fitted on each respective series 
# (naive actually doesn't require any fitting)
class Oracle(pl.LightningModule):
    def __init__(self, method="naive", use_teacher_forcing=False):
        super().__init__()
        self.method = method
        self.use_teacher_forcing = use_teacher_forcing

    def naive_forecast(self, X):
        use_teacher_forcing = True if np.random.random() < self.use_teacher_forcing else False
        if use_teacher_forcing:
            y_hat = X[:, :, -1]
        else:
            y_hat = torch.repeat_interleave(X[:, 0, -1].unsqueeze(-1), X.shape[1], dim=-1)
        
        return y_hat.unsqueeze(-1) # BS x Forecast horizon (ts len) x TS dim

    def ses_forecast(self, X):
        y_hat = []
        use_teacher_forcing = True if np.random.random() < self.use_teacher_forcing else False
        for ts in X:
            curr_pred=[]
            if use_teacher_forcing:
                # Apply teacher forcing -> fit to each window and perform a one-step ahead rolling forecast
                for x in ts:
                    ses = SimpleExpSmoothing(x.numpy()).fit(smoothing_level=0.3, optimized=True)
                    curr_pred.append(ses.forecast(1))
            else:
                # No teacher forcing, fit only on the first training sequence values and apply a forecast until the end
                ses = SimpleExpSmoothing(ts[0].numpy()).fit(smoothing_level=0.3, optimized=True)
                curr_pred.append(ses.forecast(X.shape[1]))

            y_hat.append(curr_pred)

        return torch.tensor(np.array(y_hat))
    
    def holt_forecast(self, X):
        y_hat = []
        use_teacher_forcing = True if np.random.random() < self.use_teacher_forcing else False
        for ts in X:
            curr_pred=[]
            if use_teacher_forcing:
                # Apply teacher forcing -> fit to each window and perform a one-step ahead rolling forecast
                for x in ts:
                    holt = Holt(x.numpy()).fit()
                    curr_pred.append(holt.forecast(1))
            else:
                # No teacher forcing, fit only on the first training sequence values and apply a rolling forecast until the end
                holt = Holt(ts[0].numpy()).fit()
                curr_pred.append(holt.forecast(X.shape[1]))

            y_hat.append(curr_pred)

        return torch.tensor(np.array(y_hat))
    
    def forward(self, X):
        y_hat = None
        if self.method == "naive":
            y_hat = self.naive_forecast(X)
        elif self.method == "ses":
            y_hat = self.ses_forecast(X)
        else:
            y_hat = self.holt_forecast(X)
        
        return y_hat