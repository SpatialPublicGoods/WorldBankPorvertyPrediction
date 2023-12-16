from sklearn.model_selection import BaseCrossValidator
import numpy as np

class CustomTimePanelSplit(BaseCrossValidator):
    """
    Custom cross-validator for panel data that splits the data into training and testing 
    sets based on a specified point in time. All observations before the time point are 
    used for training, and observations after are used for testing.
    """
    def __init__(self, time_column, split_time):
        self.time_column = time_column
        self.split_time = split_time

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        """
        # Sort X by time_column
        X_sorted = X.sort_values(self.time_column)
        
        # Find the index where the split should occur
        split_index = X_sorted[X_sorted[self.time_column] <= self.split_time].shape[0]

        # Yield the indices for train and test sets
        train_indices = np.arange(0, split_index)
        test_indices = np.arange(split_index, X.shape[0])
        yield train_indices, test_indices

    def get_n_splits(self, X, y, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator
        """
        return 1
