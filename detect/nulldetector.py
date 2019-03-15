import pandas as pd

from .detector import Detector
from utils import NULL_REPR


class NullDetector(Detector):
    """
    An error detector that treats null values as errors.
    """

    def __init__(self, limit_to_attrs=[], name='NullDetector'):
        super(NullDetector, self).__init__(name)
        self.limit_to_attrs = limit_to_attrs

    def setup(self, dataset, env):
        self.ds = dataset
        self.env = env
        self.df = self.ds.get_raw_data()
        if len(self.limit_to_attrs) == 0:
            self.limit_to_attrs = self.ds.get_attributes()

    def detect_noisy_cells(self):
        """
        detect_noisy_cells returns a pandas.DataFrame containing all cells with
        NULL values.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute with NULL value for this entity
        """
        errors = []
        for attr in self.limit_to_attrs:
            tmp_df = self.df[self.df[attr] == NULL_REPR]['_tid_'].to_frame()
            tmp_df.insert(1, "attribute", attr)
            errors.append(tmp_df)
        errors_df = pd.concat(errors, ignore_index=True)
        return errors_df

