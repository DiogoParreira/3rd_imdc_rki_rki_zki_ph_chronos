import torch 
import pandas as pd 
from tqdm import tqdm
import numpy as np
from typing import List, Optional
from chronos import Chronos2Pipeline
from pathlib import Path
import json

from .predictions import Predictions
from .basemodel import BaseModel

class ChronosModel(BaseModel):
    """ 
    Chronos-2 model
    Subclass of BaseModel

    Based on Amazon's Chronos-2 (model id ``amazon/chronos-2``), accessed via
    the ``chronos-forecasting`` package (``pip install "chronos-forecasting>=2.0"``)

    See the Github and Hugging Face pages for more information at
    https://github.com/amazon-science/chronos-forecasting
    https://huggingface.co/amazon/chronos-2

    Model weights
    -------------
    Unlike the older Chronos / Chronos-Bolt models on the Hugging Face Hub,
    Chronos-2 is distributed via an S3 bucket with a CloudFront CDN (boto3
    handles the download and local caching). The weights are NOT committed to
    this repo.

    The weights for this project were downloaded once in May 2026 and saved
    locally to the directory referenced by `self.pathmanager.model_utils`,
    which holds `config.json`, `.gitattributes` and `model.safetensors`.
    To reproduce, run once:

        from chronos import Chronos2Pipeline
        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        pipeline.save_pretrained("[path matching pathmanager.model_utils]")

    Alternatively, pass `"amazon/chronos-2"` to `from_pretrained` directly
    and let the library download and cache the weights automatically.

    Parameters
    ----------
    past_exogenous: Optional[List[str]] = None
        a list of column names (str) that may be used as exogenous variables in the past (alongside context data)
    future_exogenous: Optional[List[str]] = None
        a list of column names (str) that may be used as exogenous variables in the future

    See Also
    --------
    For more information, please see BaseModel
    
    Examples
    --------
    >>> chronos_univ = ChronosModel()
    >>> chronos_univ.forecast()
    >>> chronos_univ.plot_predictions(1,'national')
    >>> chronos_univ.save_predictions()    
    """
    MODEL_ID = "amazon/chronos-2"  # canonical source if no local copy is present

    quantiles_map = {
        10: 'pred',
        
        5:  'lower_50',  # 0.25
        15: 'upper_50',  # 0.75

        2:  'lower_80',  # 0.10
        18: 'upper_80',  # 0.90

        1:  'lower_90',  # 0.05
        19: 'upper_90',  # 0.95

        0:  'lower_95',  # 0.01
        20: 'upper_95',  # 0.99
        }        

    def __init__(self, 
                 past_exogenous:    Optional[List[str]] = None,
                 future_exogenous:  Optional[List[str]] = None):
        
        modelcolor                  = 'orange'
        self.past_exogenous         = past_exogenous
        self.future_exogenous       = future_exogenous        

        if past_exogenous is None and future_exogenous is None:
            modelname       = 'chronos_univariate'

        else:
            modelname       = 'chronos_multivariate'

        if future_exogenous is not None:
            raise ValueError('currently future_exogenous is not supported! Gap makes it difficult...')
        
        super().__init__(modelcolor, modelname)

        # init model: needs respective config.json, gitattributes and model.safetensors
        self.model: Chronos2Pipeline = Chronos2Pipeline.from_pretrained(
            self._resolve_model_source(),
            device_map="cpu",
        )

    def _resolve_model_source(self) -> str:
        """
        Return where to load Chronos-2 weights from.

        Uses the local directory at `pathmanager.model_utils` if it contains
        the model weights (`model.safetensors`); otherwise falls back to the
        hub id `amazon/chronos-2`, which the chronos-forecasting library
        downloads and caches automatically.
        """
        local = Path(self.pathmanager.model_utils)
        if (local / "model.safetensors").is_file():
            return str(local)
        return self.MODEL_ID

    def forecast(self):
        """
        forecast data based on contextdata and testdata. 
        The latter is only used to determine the length of predictions, or for the exogenous variables.
        predictions are stored in `._preds` and may be acessed from `.predictions`.
        """
        preds           = {}
        datasets        = self.datasets.get_list_datasets()

        for id in datasets:
            train, test = self.datasets.get_data_id(id)
            
            # date-handling
            dates       = list(test[self.col_date].unique()) # dates of validation data
            train_max   = train['date'].max()
            test_min    = test['date'].min()
            gap_dates   = pd.date_range(start=train_max,end=test_min,freq="W").tolist()[1:-1] # exclulde first (final date of training data) and last (first date of validation data)

            train_groups        = train.groupby(self.col_node)  # group by state
            test_groups         = test.groupby(self.col_node)   # group by state
            n_states            = len(train_groups)

            pred_dates          = gap_dates + dates
            df_id               = []                            # each state gets its own predictions

            for uf_code, train_group in tqdm(
                train_groups,
                total=n_states,
                desc=f"states for dataset {id}"
            ):
                test_group = test_groups.get_group(uf_code)

                train_target = torch.tensor(
                        train_group[self.col_target].values,
                        dtype=torch.float32
                    )   
                
                # inputs expected for chronos is a list, with dictionary that has keys 'target','past_covariates' (optional) and 'future_covariates' (optional).
                inputs = [{
                    "target"            : train_target,
                }]

                if self.past_exogenous is not None:
                    # produce a dict of variable-name: torch.tensor of values, same shape as context-target for each variable
                    past_covariates = {
                        covariate: torch.tensor(
                            train_group[covariate].values,
                            dtype=torch.float32) 
                            for covariate in self.past_exogenous
                    }
                    inputs[0]["past_covariates"] = past_covariates # type: ignore

                if self.future_exogenous is not None:
                    # produce a dict of variable-name: torch.tensor of values, same shape as context-target for each variable                    
                    future_covariates = {
                        covariate: torch.tensor(
                            test_group[covariate].values,
                            dtype=torch.float32) 
                            for covariate in self.future_exogenous
                    }         
                    inputs[0]["future_covariates"] = future_covariates # type: ignore

                pred_steps = len(pred_dates)

                raw_predictions = self.model.predict(
                    inputs,
                    prediction_length=pred_steps,
                )[0].squeeze(0)                     # shape [21, pred_steps]

                df_id.append(
                    self._reshape_preds(
                        raw_predictions,
                        uf_code, # type: ignore
                        pred_dates
                    )
                )

            preds[id] = pd.concat(df_id, ignore_index=True)
        self._preds = Predictions(preds)

    def _reshape_preds(
        self,
        raw_preds:  torch.Tensor,
        uf_code:    int,
        dates:      List[pd.Timestamp]
    ):
        """reshape chronos output to a more pleasant dataframe."""
        #get numpy array
        preds_np        = raw_preds.detach().cpu().numpy()

        # get right quantiles:
        quantiles_idx       = list(self.quantiles_map.keys())
        quantiles_lbls      = list(self.quantiles_map.values())

        preds_np_selection  = preds_np[quantiles_idx]

        # transform to pandas
        df = pd.DataFrame(
            preds_np_selection.transpose(),
            columns=quantiles_lbls 
        )

        df[self.col_node] = uf_code
        df[self.col_date]  = dates

        df = df[[self.col_node, self.col_date] + self.cols_prediction]

        return df   
