from django.shortcuts import render,redirect
from django.http import HttpResponse 
from django.contrib import messages
from .models import Users,MyImage
from .forms import SignupForm,LoginForm,ImageUploadForm
from django.contrib.auth.models import User,auth
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from PIL import Image
import numpy as np
from keras.models import load_model
import os
from . import MaizeML,maizeResnet50,maizeAlexnet


def index(request): 
   
    if request.method =='POST':
        Email = request.POST.get('Email')
        context= {"email":Email}
        return render(request,'Maize/register.html',context)
    else:
        return render(request,'Maize/landing.html')


def register(request):
    form = SignupForm()
    if request.method=='POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            username= request.POST["Email"]
            password = request.POST["password"]
            first_name=request.POST["First_Name"]
            
            if Users.objects.filter(Email=username).exists():
                messages.info(request,'Email already Existed!')
                
                return redirect('register')
            else:
                user=User.objects.create_user(username=username,password=password,first_name=first_name)
                user.save()
                form.save()
            return redirect('login')
        else:
            return render(request, 'Maize/my_form.html', {'form': form, 'errors': form.errors})
    context ={'form':form}
    return render(request,'Maize/register.html',context)


def login_page(request):
    form=LoginForm()
    if request.method =='POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['Email']
            password = form.cleaned_data['password']
            user=auth.authenticate(request,username=username,password=password)
            if user is not None:
                login(request, user)
                request.session['authenticated'] = True
                return redirect('home')
            else:
                messages.info(request,'Invalid Email or Password!')
                return redirect('login')
        else:
            messages.info(request,'form is not valid')
            return redirect('login')
    else:
        context ={'form': form}
        return render(request,'Maize/login.html',context)


def logout_view(request):
    logout(request)
    return redirect('login')

from .models import MyImage  # Import the MyImage model from your models.py file

def predict(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Delete the existing image if it exists
            existing_image = MyImage.objects.first()
            if existing_image:
                existing_image.corn_Image.delete()

            # Save the new uploaded image
            image = form.cleaned_data['corn_Image']
            my_image = MyImage(corn_Image=image)
            my_image.save()

            # Process the image for prediction
            image_file = my_image.corn_Image.path  # Get the path to the saved image
            img = Image.open(image_file)
            img = img.resize((50, 50))
            img = np.array(img)
            img = img.reshape(1, 50, 50, 3)

            # Load the model
            vgg16_model = load_model('MaizeML.h5')
            resnet50_model = load_model('maizeResnet50.h5')
            alexnet_model = load_model('maizeAlexnet.h5') 

            # Make the prediction
            vgg16_prediction = vgg16_model.predict(img)
            resnet50_prediction = resnet50_model.predict(img)
            alexnet_prediction = alexnet_model.predict(img)
            vgg16_class_index  = np.argmax(vgg16_prediction)
            resnet50_class_index = np.argmax(resnet50_prediction)
            alexnet_class_index = np.argmax(alexnet_prediction)
            models = ['VGG16', 'ResNet50', 'AlexNet']
            accuracies = [vgg16_prediction[0][vgg16_class_index],resnet50_prediction[0][resnet50_class_index],alexnet_prediction[0][alexnet_class_index]]
            highest_accuracy_index = np.argmax(accuracies)
            best_model = models[highest_accuracy_index]
            best_accuracy = accuracies[highest_accuracy_index]
            # class_names = [ 0,1,2,3]
            # prediction_result = class_names[highest_accuracy_index]

            model=vgg16_class_index
            if best_model=='VGG16':
                model=vgg16_class_index
            elif best_model=='ResNet50':
                model=resnet50_class_index
            elif best_model=='AlexNet':
                model=alexnet_class_index
            # model=models[highest_accuracy_index]
            print(models[highest_accuracy_index] )
            print(accuracies)
            print(vgg16_class_index)
            print(resnet50_class_index)
            print(alexnet_class_index)
            print(vgg16_prediction)

            # res = model.predict(img)
            # classification = np.argmax(res)
            # confidence = res[0][classification] * 100
            # confidence = vgg16_prediction[0][vgg16_class_index ] * 100

            # Determine the template based on the classification
            template_name = None
            if model == 0:
                template_name = 'Maize/Maize_LeafBlight.html'
            elif model == 1:
                template_name = 'Maize/Maize_CommonRust.html'
            elif model == 2:
                template_name = 'Maize/Maize_grayLeafSpot.html'
            elif model == 3:
                template_name = 'Maize/Maize_healthy.html'

            # Render the template with the appropriate classification
            return render(request, template_name)
        else:
            print(form.errors)
            return render(request, 'Maize/my_form.html', {'form': form, 'errors': form.errors})
    else:
        form = ImageUploadForm()
    return render(request, 'Maize/Maize_upload.html', {'form': form})


# def result(request):
#     if request.method == 'POST':
#         image_file = request.FILES['image']
        
#         # Perform classification on the uploaded image
#         img = Image.open(image_file)
#         img = img.resize((224, 224))
#         img = np.array(img)
#         img = img.reshape(1, 224, 224, 3)
#         model = load_model('MaizePredict\Maize\MaizeML.py')  # Update with the path to your saved model
#         res = model.predict(img)
#         classification = np.argmax(res)
#         confidence = res[0][classification] * 100

#         template_name = None
#         if classification == 0:
#             template_name = 'Maize/Maize_LeafBlight.html'
#         elif classification == 1:
#             template_name = 'Maize/Maize_CommonRust.html'
#         elif classification == 2:
#             template_name = 'Maize/Maize_grayLeafSpot.html'
#         elif classification == 3:
#             template_name = 'Maize/Maize_healthy.html'
#         return render(request, template_name)
#     return render(request, 'Maize/predict.html')