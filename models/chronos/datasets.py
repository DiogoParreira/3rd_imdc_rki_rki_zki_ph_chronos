from dataclasses import  dataclass
from typing import Dict, Tuple, List
import pandas as pd

@dataclass
class DataSets:
    """ 
    Dataclass that captures all train and test dfs.

    Parameters
    ----------
    trains: Dict[int, pd.DataFrame]
        should be Dict[1: df, 2: df, 3: df]
    tests: Dict[int, pd.DataFrame]
        should be Dict[1: df, 2: df, 3: df]        

    Note
    ----
    this class works with 'id', i.e. an integer from 1-3.

    Methods & Attributes
    --------------------
    - `get_data_id()`
    - `get_list_datasets()`
    - 'n_datasets`
    """
    trains:Dict[int, pd.DataFrame]
    tests: Dict[int, pd.DataFrame]

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        if len(self.trains) != len(self.tests):
            raise ValueError('Unequal number of train dfs vs test dfs')
        
        if self.trains.keys() != self.tests.keys():
            raise ValueError('Unequal keys for trains and tests dicts')
        
    def get_data_id(self, id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """get train and test data for certain id (1-3)"""
        return (self.trains[id], self.tests[id])
    
    @property 
    def n_datasets(self) -> int: 
        """get number of datasets (3 if all train and test are full, 1 if there's one of each)"""
        return len(self.trains)
    
    def get_list_datasets(self) -> List[int]:
        """get list of all unique ids for datasets"""
        return list(self.trains.keys())
    
    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}({len(self.trains)} datasets at .trains and .tests)>"
        return representation