import sklearn
def precict_model(pclass,sex,age,sibsp,parch,fare,embarked,title):
    import pickle
    x=[[pclass,sex,age,sibsp,parch,fare,embarked,title]]
    randomforst=pickle.load(open('titanic_model.sav', 'rb'))
    prediction=randomforst.predict(x)
    if prediction==0:
        prediction='Not Survived'
    elif prediction==1:
        prediction='Survived'
    else:
        prediction='Prediction Error'
    return prediction