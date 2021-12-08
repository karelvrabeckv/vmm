from django import forms

class SearchForm(forms.Form):
    # search form inputs
    imageInput = forms.ImageField(label = 'Select the image:')
    saveToDBInput = forms.BooleanField(label = 'Save to database', required = False)
