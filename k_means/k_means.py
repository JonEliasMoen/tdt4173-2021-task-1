import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans(object):

    def __init__(self, myK, method="Frogy"):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        #self.centroids = np.empty(0)
        self.k = myK # number of clusters
        self.centroids = np.empty(0) # centroids
        self.method = method # initial assignment
        self.Xcent = np.empty(0) # 1-D array size=X size, centroid assignment for each X datapoint
        self.cent = [] # 2-D array(M, ..) all indexes in X for centoid i

    def assignToCenter(self, xNumb, cNumb):
        oldC = int(self.Xcent[xNumb]) # previous cluster number
        if oldC != -1: # if this is not its first assignment
            self.cent[oldC].remove(xNumb) # remove from centroid->X number datastructure
        self.Xcent[xNumb] = cNumb # update 1d list
        self.cent[cNumb].append(xNumb) # update 2d list

    def assignClosest(self, X): # assign all points to closest centroid
        for i, sample in enumerate(X): # go through all points
            dist = [] # list for distances
            for c in self.centroids: # go through all centroids
                dist.append(euclidean_distance(np.array(sample), np.array(c)))
            myC = int(np.argmin(dist)) # centroid with lowest distance
            self.assignToCenter(i, myC) # assign datapoint i to cluster myC

    def fit(self, X):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            k Integer, how many clusters to use
            random Boolean, if true selects K random samples in X
                as initial centroids
        """
        # TODO: Implement
        # X-preprocessing
        self.xMax = np.max(X[:, 0]) # save ranges.
        self.yMax = np.max(X[:, 1])
        X[:, 0] = X[:,0]/np.max(X[:,0]) # normalise
        X[:, 1] = X[:, 1]/np.max(X[:,1])
    
        # initialize centroids, either random or k-first
        if self.method == "Frogy": # frogy, choose random k observations
            indexes = np.random.randint(len(X),size=(self.k))
            self.centroids = np.array(X)[indexes]
        elif self.method == "First K": # select the first k observations
            self.centroids = np.array(X[:self.k].copy())
        # initliallize other arrays
        self.Xcent = np.zeros(X.shape[0])
        self.Xcent[:] = -1
        self.cent = [[] for i in range(self.k)]


    def upscaleCentroids(self): # upscale the resulting centroids
        self.centroids[:,0] = self.centroids[:,0]*self.xMax
        self.centroids[:,1] = self.centroids[:,1]*self.yMax
    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        dist = 2
        while dist !=  0: # while converging
            self.assignClosest(X) # assign all points to closest centroid
            dist = 0
            for i in range(self.k):
                new = [0,0]
                new[0] = np.mean(X[self.cent[i], 0]) # calculate means
                new[1] = np.mean(X[self.cent[i], 1])

                dist += euclidean_distance(self.centroids[i], new) # add change in distance
                self.centroids[i] = new # assign new centroid
        self.upscaleCentroids() # scale up centroids to original data
        return np.array(self.Xcent, np.int)

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids




# --- Some utility functions


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the raw distortion measure
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()

    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points

    Note: by passing "y=0.0", it will compute the euclidean norm

    Args:
        x, y (array<...,n>): float tensors with pairs of
            n-dimensional points

    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points

    Args:
        x (array<m,d>): float tensor with pairs of
            n-dimensional points.
        y (array<n,d>): float tensor with pairs of
            n-dimensional points. Uses y=x if y is not given.

    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance

    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))
