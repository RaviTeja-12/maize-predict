from django.db import models
from PIL import Image
# Create your models here.
class Users(models.Model):
    
    
    First_Name = models.CharField(max_length=100)
    Email = models.EmailField( max_length=254)
    password = models.CharField(max_length=20,default='pass')
    def __str__(self):
        return self.First_Name



class MyImage(models.Model):
    corn_Image = models.ImageField(upload_to='images/')

