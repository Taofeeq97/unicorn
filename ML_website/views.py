from django.shortcuts import render
from . import fake_model
from . import ML_titanic_predict

def index(request):
    return render(request,'index.html')

def result(request):
    pclass=int(request.GET['pclass'])
    sex = int(request.GET['sex'])
    age = int(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    fare = int(request.GET['fare'])
    embarked = int(request.GET['embarked'])
    title = int(request.GET['title'])
    prediction=ML_titanic_predict.precict_model(pclass,sex,age,sibsp,parch,fare,embarked,title)
    return render(request,'result.html', {'prediction':prediction})