from django.urls import path
from .views import index,login_page,register,logout_view,predict

urlpatterns = [
    path('', index, name = 'home'),
    path('login',login_page,name='login'),
    path('register',register,name='register'),
    path('logout',logout_view,name='logout'),
    path('predict',predict,name='predict'),
    # path('result',result,name='result')
]