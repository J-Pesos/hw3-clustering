import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        # Basic error handling.
        assert X.size > 0, 'You must provide a non-empty X matrix.'
        assert Y.size > 0, 'You must provide a non-empty Y matrix.'
        assert X.shape[0] == Y.shape[0], 'Matrices X and Y must have the same number of rows.'

        # Initialize list of scores.
        scores = []
        
        