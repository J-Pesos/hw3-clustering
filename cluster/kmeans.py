import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        # Defining attributes for class.
        self.k = int(k)
        self.max_iter = int(max_iter)
        self.tol = tol
        self._error_new = np.inf
        self.centroids = [[]]
        self.clusters =[]

        # Basic error handling.
        assert max_iter > 0, 'You must provide a maximum number of iterations greater than 0.'
        assert k > 0, 'You must provide a k greater than 0.'

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # Define features and observations.
        num_features = mat.shape[1]
        num_obs = mat.shape[0]
        
        # Basic error handling.
        assert num_features > 0, 'There must be features in the dataset.'
        assert num_obs > 0, 'There must be observations in the dataset.'
        #assert len(mat) > self.k, 'You can not provide a higher k than the number of observations.'

        # Intiialize random number of k from the dataset as beginning centroids.
        self._init_centroids(mat)
        print(self.centroids)

        # Initialize iterations and ensure they are under specified max.
        iterations = 0

        # Intiliaze error record.
        fit_error = np.inf

        while iterations < self.max_iter and fit_error > self.tol:
            # Assign points to clusters based on minimum distance from generated array.
            self._create_clusters(mat)
            
            # Calculate centroids for new clusters.
            self.get_centroids(mat)
            print(self.centroids)

            # Calculate error for new fit.
            self._get_error()
            fit_error = self._error_new 
            print(fit_error)

            # Increase iteration by 1.
            iterations += 1


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        # Basic error handling.
        assert mat.size > 0, 'You must provide a non-empty matrix.'
        assert mat.shape[0] >= self.k, "The number of rows in the matrix must equal or exceed k."

        self.fit(mat)
        return self.clusters

    def _get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        SSE = 0
        for point in range(self._distances.shape[0]):
            SSE = SSE + np.square(self._distances[point][self.clusters[point]])

        self._error_new = SSE/self._distances.shape[0]  # Private variable, not necessary that user accesses this.

    def get_centroids(self, mat: np.ndarray) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        centroids_new = np.zeros((self.k, mat.shape[1]))
        for cluster in range(self.k):
            points = mat[np.where(self.clusters == cluster)]
            if len(points) == 0:
                centroids_new[cluster] = self.centroids[cluster]
            else:
                centroids_new[cluster] = np.mean(points, axis = 0)
        self.centroids = centroids_new

    def _init_centroids(self, mat: np.ndarray):
        """
        Intiialize random number of k from the dataset as beginning centroids.
        """
        
        self.centroids = mat[np.random.choice(np.arange(mat.shape[0]), size=self.k, replace=False)]

    def _create_clusters(self, mat: np.ndarray):
        """
        Intiialize random number of k from the dataset as beginning centroids.
        """

        self._distances = cdist(mat, self.centroids, 'euclidean') # Private variable, not necessary that user accesses this.
        self.clusters = np.argmin(self._distances, axis = 1)