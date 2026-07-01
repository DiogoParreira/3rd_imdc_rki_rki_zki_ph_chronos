from dataclasses import dataclass 
import pandas as pd 
from typing import Dict, List

@dataclass
class Predictions:
    """ 
    Dataclass that captures all predictions; analogous to DataSets.

    Parameters
    ----------
    preds: Dict[int, pd.DataFrame]
        should be Dict[1: df, 2: df, 3: df]

    Methods & Attributes
    --------------------
    - `get_preds_id()`
    - `get_list_datasets()`
    - 'n_datasets`
    """
        
    preds: Dict[int, pd.DataFrame]
        
    def get_preds_id(self, id: int) -> pd.DataFrame:
        """get preds"""
        return self.preds[id]
    
    @property 
    def n_datasets(self) -> int: 
        """get number of datasets"""
        return len(self.preds)
    
    def get_list_datasets(self) -> List[int]:
        """get list of all unique ids for preds"""
        return list(self.preds.keys())
    
    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}({len(self.preds)} preds at .preds)>"
        return representation
