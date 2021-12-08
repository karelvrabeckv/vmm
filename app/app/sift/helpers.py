import numpy as np
import cv2 as cv

import json
from json import JSONEncoder
from numpy import dot
from numpy.linalg import norm

# encrypt keypoints to array
# https://stackoverflow.com/questions/26501193/opencv-python-find-a-code-for-writing-keypoins-to-a-file
def keypointsEncoder(keypoints):
  keypointsArray = []
  for point in keypoints:
    temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id)
    keypointsArray.append(temp)

  return keypointsArray

def keypointsDecoder(keypoints):
  kp = []
  for point in keypoints:
    temp = cv.KeyPoint(x=point[0][0],y=point[0][1], size=point[1], angle=point[2],
                            response=point[3], octave=point[4], class_id=point[5])
    kp.append(temp)
  return kp


# https://pynative.com/python-serialize-numpy-ndarray-into-json/
class NumpyArrayEncoder(JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return JSONEncoder.default(self, obj)

def NumpyArrayDecoder(JSONString, type = "object"):
  decodedArrays = json.loads(JSONString)
  return np.asarray(decodedArrays["array"], type)

# cosinus distance between 2 vectors
def cosDistance(vector1, vector2):
  return dot(vector1, vector2) / (norm(vector1) * norm(vector2))