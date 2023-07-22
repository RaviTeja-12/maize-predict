from django import forms
from .models import Users,MyImage

class SignupForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    
    class Meta:
        model = Users
        fields = '__all__'

class LoginForm(forms.Form):
    Email = forms.EmailField(max_length=100)
    password = forms.CharField(widget=forms.PasswordInput)

class ImageUploadForm(forms.Form):
    corn_Image = forms.ImageField(required=True)