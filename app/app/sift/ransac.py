import numpy as np
import random

# class for calculating homography matches of descriptor by RANSAC
# require numpy array with all matched vectors and optionally number of iterations
class Ransac():
  matches       = np.array([])  # original array with matches (index of keypoint vs index of keypoint)
  keypointsImg1 = np.array([])  # original array with keypoints for img1
  keypointsImg2 = np.array([])  # original array with keypoints for img2
  bestResult    = 0             # best result - maximum of descriptor in delta surroundings

  # require numpy array with all matched matches and optionally number of iterations
  def __init__(self, matches, keypointsImg1, keypointsImg2, maxIterations = 100, threshold = 0.5):
    # save matches and keypoints
    self.matches = np.asarray(matches)
    self.keypointsImg1 = keypointsImg1
    self.keypointsImg2 = keypointsImg2

    for ite in range(0, maxIterations):
      # select 4 match pairs
      randomIndexes = random.sample(range(len(self.matches)), 4)

      # get random two vectors
      selectedMatches = np.array([self.matches[randomIndexes[0]], self.matches[randomIndexes[1]], self.matches[randomIndexes[2]], self.matches[randomIndexes[3]]])

      # calculate homography matrix
      matrix = self.homographyMatrix(selectedMatches)

      #  if rank is lower than 3, continue to prevent dividing by zero
      if np.linalg.matrix_rank(matrix) < 3:
          continue

      distances = self.getViewsCoordsDistance(self.matches, matrix)
      indexes = np.where(distances < threshold)[0]
      inliers = self.matches[indexes]

      # calculate percentage
      result = (100 * len(inliers))/len(self.matches)

      if self.bestResult < result:
        self.bestResult = result

      # if image is same
      if result >= 100:
        break

  # return best result (in percentage) of vectors in delta surroundings
  def getBestResult(self):
    return self.bestResult

  # make transformation matrix
  def homographyMatrix(self, selectedMatches):
      rows = []
      for i in range(selectedMatches.shape[0]):
          p1 = self.keypointsImg1[selectedMatches[i][0]].pt
          p2 = self.keypointsImg2[selectedMatches[i][1]].pt

          # append rows to matrix - http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf slide 8
          rows.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]])
          rows.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]])

      rows = np.array(rows)
      # calculate values to transformation matrix (to V)
      U, s, V = np.linalg.svd(rows)
      # transform results (9x1) to 3x3 matrix
      H = V[-1].reshape(3, 3)
      # standardize - set 1 to 3,3 element
      H = H/H[2, 2]
      return H

  # return array with distance between expected position of points and real position
  def getViewsCoordsDistance(self, matches, matrix):
      matchesNum = len(matches)

      # get coordinates all points
      keypointsCoordsImg1 = np.array([
        [self.keypointsImg1[indexes[0]].pt[0], self.keypointsImg1[indexes[0]].pt[1], 1] for indexes in matches
      ])
      keypointsCoordsImg2 = np.array([
        [self.keypointsImg2[indexes[1]].pt[0], self.keypointsImg2[indexes[1]].pt[1]] for indexes in matches
      ])

      # make array with matches size filled with zero, where should be a result
      viewCoords = np.zeros((matchesNum, 2))

      for i in range(matchesNum):
          # calculate coordinates of view for point from keypoints IMG 1
          viewCoordsTmp = np.dot(matrix, keypointsCoordsImg1[i])
          viewCoords[i] = (viewCoordsTmp/viewCoordsTmp[2])[0:2]

      # compute distance from points in 2 from their "view" place
      img2DistancesFromView = np.linalg.norm(keypointsCoordsImg2 - viewCoords , axis=1) ** 2

      return img2DistancesFromView