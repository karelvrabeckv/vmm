import numpy as np
from numpy import linalg
import random
import datetime
import math
import time as t

# class for calculating k-clusters with vectors
# require numpy array with all vectors and optionally existing centroids
class Kmeans():
  k                 = 2000         # number of clusters
  clusters          = []            # multidimensional array with final clusters - array with vector's index in numpyArray type
  clustersDesIndex  = []            # multidimensional array with final clusters - array with vector's index in numpyArray type
  centroids         = np.array([])  # array with centroid's vectors (key = index/id cluster; value = centroid vector)
  vectors           = np.array([])  # original array with vectors

  # require numpy array with all vectors and optionally existing centroids
  def __init__(self, vectors, maxIterations = 9999, centroids = []):
    # save vectors
    self.vectors = vectors

    # if centroids is not set - we will calculate visual word for database
    if len(centroids) == 0:
      for ite in range(0, maxIterations):
        # get old centroids for compare old and new - if its same, k-means can break loop
        oldCentroids = self.centroids

        # make new centroids
        self.makeCentroids()

        # check, if new centroids is different from old
        if np.array_equal(oldCentroids, self.centroids) == True:
          break

        # calculate clusters
        self.makeClusters()
    else:
      # calculate clusters (histogram) from existing BoF
      self.centroids = centroids
      self.makeClusters()

  # get histogram calculated for clusters
  def getHistogram(self):
    histogram = np.array([])
    for cluster in self.clusters:
      histogram = np.append(histogram, len(cluster))

    return histogram

  # get centroids
  def getCentroids(self):
    return self.centroids

  # get clusters with descriptors
  def getClusters(self):
    return self.clusters

  # get clusters with descriptors index
  def getClustersDesIndex(self):
    return self.clustersDesIndex

  def makeClusters(self):
    self.clusters         = [[] for _ in self.centroids]
    self.clustersDesIndex = [[] for _ in self.centroids]

    vectors = self.vectors.astype(np.float64)
    centroids = self.centroids.astype(np.float64)

    for vectorID, vector in enumerate(vectors):
        distances = np.linalg.norm(vector - centroids, axis = 1)
        index = np.argmin(distances)
        self.clusters[index].append(vector)
        self.clustersDesIndex[index].append(vectorID)

  # pick centroids - for first iteration make random centroids
  def makeCentroids(self):
    if len(self.centroids) == 0:
      # make random centroid for each cluster
      indexes = random.sample(range(len(self.vectors)), min(self.k, len(self.vectors)))

      for i in indexes:
        # centroids found - save
        if len(self.centroids) == 0:
          self.centroids = np.array([self.vectors[i]])
        else:
          self.centroids = np.append(self.centroids, [self.vectors[i]], axis = 0)
    else:
      # generate new centroids from average from all vectors in cluster for each component in vector
      for centroidID, centroid in enumerate(self.centroids):
        self.centroids[centroidID] = np.average(self.clusters[centroidID], axis=0)