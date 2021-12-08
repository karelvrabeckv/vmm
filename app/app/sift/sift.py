import numpy as np
import cv2 as cv
import time as t

import json
from .kmeans import Kmeans
from .ransac import Ransac
from django.apps import apps
from .helpers import keypointsEncoder
from .helpers import keypointsDecoder
from .helpers import NumpyArrayEncoder
from .helpers import NumpyArrayDecoder
from .helpers import cosDistance
from .knn import get_matches

# extract keypoints and descriptors from image and save to JSONField in requestImageModel
def extractKeypoints(requestImageModel, image):
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

  sift = cv.SIFT_create()
  kp, des = sift.detectAndCompute(gray,None)

  start = t.time()
  # get BoF object for getting - histogram in BoF and descriptors in cluster
  bofHistogram, bofClusters = getImageBoFData(des)
  end = t.time()
  elapsed1 = end - start # v ms

  # save to model
  numpyData = {"array": bofClusters}
  requestImageModel.bofClusters = json.dumps(numpyData, cls=NumpyArrayEncoder)
  numpyData = {"array": bofHistogram}
  requestImageModel.bofHistogram = json.dumps(numpyData, cls=NumpyArrayEncoder)

  requestImageModel.keypoints = keypointsEncoder(kp)

  # Serialization descriptors (numpy array)
  numpyData = {"array": des}
  requestImageModel.descriptors = json.dumps(numpyData, cls=NumpyArrayEncoder)

# make new BoF (after saving new image to DB)
def makeBoF(ImageModel):
  # array with all descriptors from DB
  allDescriptors = 0

  for imageModel in ImageModel.objects.all():
    if type(allDescriptors) is int:
      allDescriptors = NumpyArrayDecoder(imageModel.descriptors)
    else:
      allDescriptors = np.concatenate((allDescriptors, NumpyArrayDecoder(imageModel.descriptors)))

  # make new BoF from kmeans and save to DB
  kmeans = Kmeans(allDescriptors)

  bof = apps.get_model('sift.BackOfFeatures').objects.first()

  # if object with BoF not exists already
  if bof == None:
    bof = apps.get_model('sift.BackOfFeatures')()

  # set BoF data to DB model
  numpyData = {"array": kmeans.getCentroids()}
  bof.centroids     = json.dumps(numpyData, cls=NumpyArrayEncoder)
  bof.save()

  # recalculate histograms and clusters data for each saved image
  for imageModel in ImageModel.objects.all():
    bofHistogram, bofClusters = getImageBoFData(NumpyArrayDecoder(imageModel.descriptors))
    numpyData = {"array": bofClusters}
    imageModel.bofClusters = json.dumps(numpyData, cls=NumpyArrayEncoder)
    numpyData = {"array": bofHistogram}
    imageModel.bofHistogram = json.dumps(numpyData, cls=NumpyArrayEncoder)
    imageModel.save()

# get histogram (BoF) and descriptors in clusters for image
def getImageBoFData(descriptors):
  bof = apps.get_model('sift.BackOfFeatures').objects.first()

  if bof == None:
    return 0,0

  # cluster image's descriptors
  kmeans    = Kmeans(descriptors, 0, NumpyArrayDecoder(bof.centroids))
  return kmeans.getHistogram(), kmeans.getClustersDesIndex()

def compareImageBoF(requestImageModel, imageModel):
  # compare histograms (BoF) of both image
  return cosDistance(NumpyArrayDecoder(requestImageModel.bofHistogram, "float32"), NumpyArrayDecoder(imageModel.bofHistogram, "float32"))

# compare two images by our code
def compareImageKNNRansac(requestImageModel, imageModel):
  # geometric verification
  # match by KNN alghoritm
  matches = get_matches(NumpyArrayDecoder(requestImageModel.bofClusters), NumpyArrayDecoder(requestImageModel.descriptors, "float32"), NumpyArrayDecoder(imageModel.bofClusters), NumpyArrayDecoder(imageModel.descriptors, "float32"))

  # RANSAC alghoritm
  ransac = Ransac(matches, keypointsDecoder(requestImageModel.keypoints), keypointsDecoder(imageModel.keypoints))

  return ransac.getBestResult()

  return 50

# compare two images by openCV code - testing purpose
def compareImageSift(requestImageModel, imageModel):
  #feature matching
  bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

  des1 = NumpyArrayDecoder(requestImageModel.descriptors).astype(np.float32)
  des2 = NumpyArrayDecoder(imageModel.descriptors).astype(np.float32)

  matches = bf.match(des1, des2)

  keypoints1 = keypointsDecoder(requestImageModel.keypoints)
  keypoints2 = keypointsDecoder(requestImageModel.keypoints)

  similarity1 = (100*len(matches)/len(keypoints1))
  similarity2 = (100*len(matches)/len(keypoints2))

  return round(max(similarity1, similarity2), 2)