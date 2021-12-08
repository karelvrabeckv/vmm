from django.contrib import admin

from .models import ImageModel, RequestImageModel

# Register your models here.
admin.site.register(ImageModel)
admin.site.register(RequestImageModel)
