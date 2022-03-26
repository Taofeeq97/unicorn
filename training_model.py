import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',13)

data_train=pd.read_csv('train.csv')


S_Sur=pd.crosstab(data_train['Survived'],data_train['Sex'])



data_train['Age'].fillna(data_train['Age'].median(), inplace=True)



Embark=data_train['Embarked'].value_counts()


data_train['Embarked'].fillna('S', inplace=True)


del data_train['Cabin']


names=data_train['Name'].sample(25)

def get_title(name):
    if ',' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'No title seen'

titles=set([name for name in data_train['Name'].map(lambda name:get_title(name))])


def comprehensive_title(x):
    title=x['Titles']
    if title in ['Capt','Maj','Col','Major']:
        return 'Military'
    elif title in ['Jonkheer','Don', 'the Countess','Lady','Sir']:
        return 'Royalty'
    elif title in ['Miss', 'Mme','Ms','Mr','Mrs']:
        return 'Individual'
    else:
        return title

data_train['Titles']=data_train['Name'].map(lambda name: get_title(name))

data_train['Title']=data_train.apply(comprehensive_title, axis=1)



data_train['Sex'].replace(('male','female'),(0,1), inplace=True)
data_train['Embarked'].replace(('S','C','Q'),(0,1,2), inplace=True)
data_train['Title'].replace(('Individual','Master','Dr', 'Rev','Royalty','Military','Mlle'),(0,1,2,3,4,5,6), inplace=True)
data_train.drop('Name', axis=1, inplace=True)
data_train.drop('Ticket', axis=1, inplace=True)
data_train.drop('Titles', axis=1, inplace=True)
data_train.drop('PassengerId', axis=1, inplace=True)




y=data_train['Survived']
x=data_train.drop('Survived',axis=1)


corr= data_train.corr()


x=data_train.drop(['Survived'], axis=1)
y=data_train['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,)
randomforst=RandomForestClassifier()
randomforst.fit(x_train,y_train)
y_pred =randomforst.predict(x_test)
acc_randomforest=round(accuracy_score(y_pred,y_test)*100,2)

pickle.dump(randomforst,open('titanic_model.sav','wb'))

test_data=pd.read_csv('test.csv')
test_data['Titles']=test_data['Name'].map(lambda name: get_title(name))

test_data['Title']=test_data.apply(comprehensive_title, axis=1)

test_data['Age'].fillna(data_train['Age'].median(), inplace=True)
test_data['Fare'].fillna(data_train['Fare'].median(), inplace=True)


ids=test_data['PassengerId']
test_data['Sex'].replace(('male','female'),(0,1), inplace=True)
test_data['Embarked'].replace(('S','C','Q'),(0,1,2), inplace=True)
test_data['Title'].replace(('Individual','Master','Dr', 'Rev','Royalty','Military','Mlle','Dona'),(0,1,2,3,4,5,6,7), inplace=True)
test_data.drop('Name', axis=1, inplace=True)
test_data.drop('Ticket', axis=1, inplace=True)
test_data.drop('Titles', axis=1, inplace=True)
test_data.drop('PassengerId', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

prediction=randomforst.predict(test_data)
output=pd.DataFrame({'PassengerId':ids, 'Survived':prediction})
output.to_csv('Submission.csv', index=False)

def precict_model(pclass,sex,age,sibsp,parch,fare,embarked,title):
    import pickle
    x=[[pclass,sex,age,sibsp,parch,fare,embarked,title]]
    randomforst=pickle.load(open('titanic_model.sav','rb'))
    predictions=randomforst.predict(x)












