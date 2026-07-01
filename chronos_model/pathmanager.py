import os 
from typing import Literal
from pathlib import Path
import sys 

class InvalidPath(Exception):
    """To be raised when path doesnt exist"""
    def __init__(self,
                 path: str):
        
        message = f'path {path} does not seem to exist.'
        super().__init__(message)
        
class PathManager:

    """
    Manager class that contains all relevant paths

    Attributes:
    - `project_root`
    - `chronos`
    - `results`
    """

    def __init__(self):

        self.project_root   = Path(__file__).resolve().parents[1]  
        self.model_utils    = os.path.join(self.project_root, 'chronos_model', 'utils')
        self.results        = os.path.join(self.project_root, 'submissions')
    
        for path in [self.project_root, self.model_utils, self.results]:
            self._validate_existence_path(path)
    
    def _validate_existence_path(self, path: str):
        if not Path(path).exists():
            raise InvalidPath(path)
        
    def __repr__(self) -> str:
        representation = "<PathManager> with attribtues project_root, model_utils, results"
        return representation