from django.db import models
from .sift import extractKeypoints
from .sift import compareImageSift
from .sift import compareImageBoF
from .sift import compareImageKNNRansac
import numpy as np
from PIL import Image

# static data for database
class BackOfFeatures(models.Model):
  centroids     = models.JSONField()

# Model for image in search process
class RequestImageModel(models.Model):
    title       = models.CharField(max_length=120)
    image       = models.ImageField()
    keypoints   = models.JSONField(blank=True, null=True)
    descriptors = models.JSONField(blank=True, null=True)
    bofClusters   = models.JSONField(blank=True, null=True)
    bofHistogram  = models.JSONField(blank=True, null=True)

    # constructor for result model class
    @classmethod
    def create(self, title, image):
        requestImageModel = self(title = title, image = image)
        return requestImageModel

    # extract keypoint by SIFT to image attribute
    def extractKeypoints(self):
        # open image
        pil_img = Image.open(self.image.open())

        # convert image to array
        cv_img = np.array(pil_img)
        extractKeypoints(self, cv_img)

    def compareImageBoF(self, imageModel):
        return compareImageBoF(self, imageModel)

    def compareImageKNNRansac(self, imageModel):
        return compareImageKNNRansac(self, imageModel)

    def _str_(self):
        return self.title

    # skip save - we don't want to save request/response model
    def save(self, *args, **kwargs):
        pass

    class Meta:
        managed = False

# Model for image in DB
class ImageModel(models.Model):
    title       = models.CharField(max_length=120)
    image       = models.ImageField(upload_to='media')
    keypoints   = models.JSONField()
    descriptors = models.JSONField()
    bofClusters   = models.JSONField()
    bofHistogram  = models.JSONField()
    created     = models.DateTimeField(auto_now_add=True)

    # constructor for result model class
    @classmethod
    def create(self, title, image, keypoints, descriptors, bofClusters, bofHistogram):
        imageModel = self(title = title, image = image, keypoints = keypoints, descriptors = descriptors, bofClusters = bofClusters, bofHistogram = bofHistogram)
        return imageModel

    def _str_(self):
        return self.title