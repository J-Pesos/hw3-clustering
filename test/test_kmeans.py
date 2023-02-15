# Write your k-means unit tests here
import numpy as np
from cluster import KMeans, make_clusters
import pytest

def test_k():
    '''
    Test KMeans ability to handle exceptions and provide correct features given
    input values of k.

    '''

    # Test Kmeans ability to handle incorrect k values.

    k_15 = KMeans(k = 15) # Larger k than n observations.
    mat_10_clusters, mat_10_labels = make_clusters(n = 10)
    with pytest.raises(Exception):
        k_15.fit(mat_10_clusters)

    with pytest.raises(Exception):
        KMeans(k = 0) # K can not be 0.

    # Ensure number of clusters created = k.
    clusters, labels = make_clusters(n=1000, m=200, k=3)

    k_3 = KMeans(k = 3)
    predict = k_3.predict(clusters)

    assert len(np.unique(predict) == 3), "Incorrect number of clusters generated for given k."

def test_labels():
    '''
    Test that the size of fit labels are the same size as predicted labels.

    '''
    
    # Generate data.
    clusters, labels = make_clusters()

    # Create labels for fit and prediction.
    kmeans = KMeans(k = 5)
    kmeans.fit(clusters)
    predict = kmeans.predict(clusters)

    # Ensure label sizes are equal.
    assert(len(predict) == len(labels)), 'Labels for fit and labels for prediction are not the same size!'