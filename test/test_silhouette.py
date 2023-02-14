# write your silhouette score unit tests here
from cluster import Silhouette, make_clusters
from sklearn.metrics import silhouette_samples
import pytest
import numpy as np

def test_silhouette():
    """
    Test that my silhouette scoring matches sklearn's scoring method.

    """
   
    # Generate data.
    clusters, labels = make_clusters()

    # Generate scores using my method.
    scores = Silhouette().score(clusters, labels)
    
    # Generate scores using sklearn.
    sklearn_scores = silhouette_samples(clusters, labels)

    # Use np.allclose to return True if two scores array are element-wise equal.
    assert np.allclose(scores, sklearn_scores) == True