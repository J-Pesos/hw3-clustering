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
        print("Silhouette v6")
        # Basic error handling.
        assert X.size > 0, 'You must provide a non-empty X matrix.'
        assert Y.size > 0, 'You must provide a non-empty Y matrix.'
        assert X.shape[0] == Y.shape[0], 'Matrices X and Y must have the same number of rows.'

        # Initialize list of scores.
        scores = np.zeros(X.shape[0])
        #print(scores)

        # Calculate score for each observation in the matrix.
        for i,row in enumerate(X):
            inter_dist_mean = []
            row_label = Y[i]
            
            for label in np.unique(Y):

                if label == row_label:
                    points = X[np.where(np.delete(Y,i) == label)]
                    #print(points)
                    #print(np.size(points))
                    if np.size(points) == 0: # If there are no other points in label, calculate mean distance 0.
                        intra_dist_mean = 0
                    else:
                        intra_dist_mean = np.mean(cdist([row], points))
                    #print(intra_dist_mean)
                else:
                    points = X[np.where(Y == label)]
                    if np.size(points) == 0:
                        inter_dist_mean += 0
                    else:
                        inter_dist_mean += [np.mean(cdist([row], points))]
                    #print(inter_dist_mean)
                    

            min_inter_dist_mean = min(inter_dist_mean)
            score = (min_inter_dist_mean - intra_dist_mean) / max(intra_dist_mean, min_inter_dist_mean)

            scores[i] = score
            #print(scores)
        
        return scores