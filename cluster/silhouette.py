import numpy as np
from scipy.spatial.distance import cdist
import math

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
        scores = np.zeros(X.shape[0])

        # Calculate score for each observation in the matrix.
        for i,row in enumerate(X):
            inter_dist_mean = []
            row_label = Y[i]
            
            for label in np.unique(Y):

                if label == row_label:
                    indices = Y == label
                    indices[i] = False
                    intra_dist_mean = np.mean(cdist([row], X[indices]))

                else:
                    points = X[np.where(Y == label)]
                    inter_dist_mean += [np.mean(cdist([row], points))]
                    
            min_inter_dist_mean = min(inter_dist_mean)
            score = (min_inter_dist_mean - intra_dist_mean) / max(intra_dist_mean, min_inter_dist_mean)

            # Manually assign a score of 0 to mean calculations that yield NaN.
            if math.isnan(score):
                score = 0

            scores[i] = score
        
        return scores