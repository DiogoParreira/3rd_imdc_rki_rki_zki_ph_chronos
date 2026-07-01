from typing import Optional 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from pathlib import Path
from abc import abstractmethod, ABC

from .pathmanager import PathManager
from .predictions import Predictions
from .datasets import DataSets

class BaseModel(ABC):
    """ 
    Parent class to all models. 
    Centralizes the datasets and predictions.

    The loading is done through the use of DataSets, and predictions
    are stored in Predictions instances. Note that we distinguish
    predictions from formatted_predictions. The latter is solely used
    upon saving predictions (not in between, not even for plotting).

    Models, or rather childclasses of this class, should have implemented
    `forecast()` methods.

    Methods
    -------
    - `plot_predictions()`
    - `save_predictions()`

    """
    pathmanager: PathManager

    _datasets:  Optional[DataSets]          = None
    _preds:     Optional[Predictions]       = None
    _preds_formatted: Optional[Predictions] = None
    
    datasets_to_load                        = [1,2,3,4]
    # runs for each instance creation
    def __init__(self, color: str, name: str):
        self.__class__._setup_pathmanager()
        self._setup_col_attrs()
        
        self.color = color
        self.name  = name

        self._preds = None 
        self._preds_formatted = None

    def plot_predictions(self,
                         dataset_id:    int,
                         state:         str,
                         context:       bool = True):
        """
        plots model predictions

        Parameters
        ----------
        dataset_id: int
            number from 1-3: the dataset IDs
        state: str
            code of the state to plot (letters, i.e. RO). Alternatively one may write "national"
        context: bool = True
            whether or not to show the context ('train') data. By default, yes
        """

        preds       = self.predictions.get_preds_id(dataset_id)
        train, test = self.datasets.get_data_id(dataset_id)

        # aggregate data:
        states = list(train['state'].unique())

        if state == 'national':
                train_df    = train.groupby([self.col_date]).agg({self.col_target : 'sum'}).reset_index(drop = False)
                preds_df    = preds.groupby([self.col_date]).agg({col:'sum' for col in self.cols_prediction}).reset_index(drop = False)
                test_df     = test.groupby([self.col_date]).agg({self.col_target : 'sum'}).reset_index(drop = False)

        elif state in states:
                train_df    = train[train[self.col_node] == state].reset_index(drop = True)
                preds_df    = preds[preds[self.col_node] == state].reset_index(drop = True)
                test_df     = test[test[self.col_node] == state].reset_index(drop = True)

        else:
            raise ValueError(f'invalid value for state {state}. Valid values are "national" or states: {states}')
        
        test_min     = test_df[self.col_date].min()
        preds_df_gap = preds_df[preds_df[self.col_date] < test_df[self.col_date].min()]
        preds_df     = preds_df[preds_df[self.col_date] >= test_min]

        upper_bound = 'upper_95'
        center_bound= 'pred'
        lower_bound = 'lower_95'

        # title
        title = f"dataset {dataset_id} - "
        if state == 'national':
            title = title + "national"
        else:
            title = title + f"state {state}"

        # init figure

        fig, ax = plt.subplots(1,1, figsize = (15,4))

        # test line: single, darkred line
        sns.lineplot(data           = test_df, 
                    x               = self.col_date, 
                    y               = self.col_target,
                    color           = 'darkred', 
                    marker          = '.', 
                    label           = 'ground truth',
                    ax              = ax, 
                    linewidth       = 2
                    )

        # preds in gap: grey bandwith
        sns.lineplot(data           = preds_df_gap, 
                    x               = self.col_date, 
                    y               = center_bound,
                    color           = 'darkgrey', 
                    label           = 'predictions - gap',
                    ax              = ax, 
                    linewidth       = 2 
                    )   
        
        ax.fill_between(
            x       = preds_df_gap[self.col_date],
            y1      = preds_df_gap[upper_bound],
            y2      = preds_df_gap[lower_bound],
            color   = 'darkgrey',
            alpha   = 0.2
        )           

        # preds after gap: model-color (self.color) bandwith 
        sns.lineplot(data           = preds_df, 
                    x               = self.col_date, 
                    y               = center_bound,
                    color           = self.color, 
                    label           = 'predictions',
                    ax              = ax, 
                    linewidth       = 2, 
                        )        

        ax.fill_between(
            x       = preds_df[self.col_date],
            y1      = preds_df[upper_bound],
            y2      = preds_df[lower_bound],
            color   = self.color,
            alpha   = 0.2,
            label   = f'uncertainty interval'
        )        

        # Add context data
        if context:

            sns.lineplot(data           = train_df, 
                        x               = self.col_date, 
                        y               = self.col_target,
                        color           = 'darkblue', 
                        label           = 'train',
                        ax              = ax, 
                        linewidth       = 2, 
                        )        
        ax.grid()
        ax.legend()
        ax.set_title(title)
        
    def save_predictions(self):
        """
        save predictions in the `results` folder attribute of the pathmanager,
        inside the folder named the model name.
        """
        self._format_predictions()
        preds_dir = os.path.join(self.pathmanager.results,self.name)
        
        if not Path(preds_dir).exists():
            Path(preds_dir).mkdir()

        else:
            raise ValueError(f'predictions - folder already exists for {self.name}. PLease remove this first.')

        for preds_name, preds_df in self.formatted_predictions.preds.items():

            if isinstance(preds_name, int):
                raise ValueError('got key for self.formatted_predictions of type int. Should be str.')
            
            filename = preds_name + '.csv'
            preds_df.to_csv(os.path.join(preds_dir, filename), index = False)
            print(f'predictions {preds_name} saved for {self.name}')

    @classmethod
    def _setup_pathmanager(cls) -> None:
        if not hasattr(cls, 'pathmanager'):
            cls.pathmanager = PathManager()

    @classmethod
    def _load_datasets(cls) -> DataSets:
        """
        load all datasets from the datafactory, and process (fixing timestamps)
        """
        from data_catalog.data_factory import data_factory        
        trains = {}
        tests  = {}
        for idx in cls.datasets_to_load:
            id = idx
            train, test     = data_factory(id)
            trains[id]      = cls._process_data(train)
            tests[id]       = cls._process_data(test)

        return DataSets(trains, tests)
    
    @classmethod
    def _process_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        df['date']  = pd.to_datetime(df['date'])
        return df

    def _setup_col_attrs(self):
        """set some column - related attributes"""
        self.col_node           = 'state'
        self.col_date           = 'date'
        self.col_target         = 'cases'
        self.cols_prediction    = ['pred'] + ['lower_50','upper_50','lower_80','upper_80','lower_90','upper_90','lower_95','upper_95']  # final order of predictions

    @classmethod
    def get_datasets(cls):
        """returns dataset if exists. if not, dataset is loaded right here."""
        if cls._datasets is None:
            cls._datasets = cls._load_datasets()
        return cls._datasets

    @property
    def datasets(self):
        return self.__class__.get_datasets() # class method

    @property
    def predictions(self) -> Predictions:
        """returns predictions if they exist. if not, exception is thrown"""
        if self._preds is None:
            raise ValueError('predictions not found.')
        return self._preds
    
    @property
    def formatted_predictions(self) -> Predictions:
        if self._preds_formatted is None:
            raise ValueError('predictions formatted not found.')
        return self._preds_formatted

    @abstractmethod
    def forecast(self):
        pass

    def _format_predictions(self):
        """ 
        format predictions for saving: removes the predictions in the gap,
        and converts dates back to strings. Also ensures the right orde
        """
        preds = {}

        # get map of dataset id - filename
        predictions_filenames = {
            1:  'test_1_2022',
            2:  'test_2_2023',
            3:  'test_3_2024',
            4:  'test_4_2025'
        }

        for key, df in self.predictions.preds.items():

            if isinstance(key, str):
                raise ValueError(f'got key of type str from self.predictions.preds.items(), should be integers 1-3. Got {key}')
            
            elif key not in predictions_filenames.keys():
                raise ValueError(f'got unexpected key from self.predictions.preds.items() {key}. Expected any of integers 1-3.')
            
            dfc         = df.copy()

            test_df             = self.datasets.tests[key]
            mindate             = test_df[self.col_date].min()
            dfc: pd.DataFrame   = dfc[dfc[self.col_date] >= mindate].reset_index(drop=True)
            
            filename = predictions_filenames[key]

            dfc[self.col_date]   = dfc[self.col_date].dt.date
            dfc                  = dfc[self.cols_prediction + [self.col_date] + [self.col_node]]
            preds[filename]      = dfc 

        self._preds_formatted             = Predictions(preds)
