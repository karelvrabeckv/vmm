import base64
from django.http import HttpResponseRedirect
from django.shortcuts import render
import time as t

from .forms import SearchForm
from .models import RequestImageModel
from .models import ImageModel
from .sift import makeBoF

# Create your views here.

def index(request):
  return render(request, 'form.html', {'form': SearchForm()})

def results(request):
  input = 'images/input.jpg'
  results = []

  # if this is a POST request we need to process the form data
  if request.method == 'POST':
    # create a form instance and populate it with data from the request:
    form = SearchForm(request.POST, request.FILES)
    # check whether it's valid:
    if form.is_valid():
      start = t.time()

      # make new model for request
      requestImageModel = RequestImageModel.create(form.cleaned_data['imageInput'].name, form.cleaned_data['imageInput'])

      # extract image's keypoints by SIFT
      requestImageModel.extractKeypoints()

      # flag for check, if same image is already in DB
      foundSameImage = False


      imageModelsBoF = []
      # save all BoF to Images
      for index, imageModel in enumerate(ImageModel.objects.all()):
        imageModelsBoF.append([index, requestImageModel.compareImageBoF(imageModel)])

      # sort array
      imageModelsBoF.sort(reverse=True, key=lambda x:x[1])

      if len(imageModelsBoF) > 0:
        # calculate geometric verification first 5 best matches
        for i in range(min(5, len(imageModelsBoF))):
          imageModel = ImageModel.objects.all()[imageModelsBoF[i][0]]
          similarity = requestImageModel.compareImageKNNRansac(imageModel)

          results.append({
              'url' :         imageModel.image.url,
              'similarity' :  round(similarity, 2)
          })

          # if this image is already in DB, ignore save checkbox
          if round(similarity) == 100:
            foundSameImage = True

      # sort results
      results.sort(reverse=True, key=lambda x:x['similarity'])

      # if image should be saved in DB and same image not found in DB - make new model with imageModel
      if form.cleaned_data['saveToDBInput'] and foundSameImage == False:
        imageModel = ImageModel.create(form.cleaned_data['imageInput'].name, form.cleaned_data['imageInput'], requestImageModel.keypoints, requestImageModel.descriptors, requestImageModel.bofClusters, requestImageModel.bofHistogram)
        imageModel.save()
        # make new BoF - after test return to if
        makeBoF(ImageModel)

      with request.FILES['imageInput'].open("rb") as image_file:
        input = base64.b64encode(image_file.read())

      input = input.decode("utf-8")
      input = "data:image/jpg;base64," + input

      end = t.time()
      time = format(end - start, ".2f")

  return render(request, 'results.html', {'input': input, 'results': results, 'amount': len(results), 'time': time})
