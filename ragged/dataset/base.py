from abc import ABC, abstractmethod

import pandas as pd

class Dataset(ABC):
    @abstractmethod
    def to_pandas(self)->pd.DataFrame:
        pass

    @staticmethod
    def available_datasets():
        """
        List of available datasets that can be loaded
        """
        return []
