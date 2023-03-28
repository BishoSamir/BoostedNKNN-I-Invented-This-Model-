from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
import numpy as np 

pastKnowledge = { 'hiddenSize' : {} }
for i in range(266):
    pastKnowledge['hiddenSize'][i] = 0

def getProperData(X , y):
    num_cols = np.random.randint(int(X.shape[1]**0.5) , X.shape[1]+1)
    # check if size = all features perform better  , later update it again to compare them against each other 
    #random_cols = np.random.choice(X.shape[1] , size = num_cols , replace = False)
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= 0.2 , random_state= 44  )
    return X_train , X_test , y_train , y_test

def getProperArch():
    hiddenSize = np.random.randint(4 , 129)
    while(pastKnowledge['hiddenSize'][hiddenSize]):
        hiddenSize = np.random.randint(4 , 129)

    pastKnowledge['hiddenSize'][hiddenSize] = 1
    return hiddenSize

def initModels(nModels):
    models = {}
    for i in range(nModels):
        models[i] = {}
    return models

def run(nModels ,X , y):
    models = initModels(nModels)    
    
    for i in range(nModels):
        X_train , X_test , y_train , y_test = getProperData(X,y)
        K = np.random.randint( 1 , min(X.shape[0] , 10)+1 )
        hiddenSize = getProperArch()
    
        models[i]['data'] = [X_train , X_test , y_train , y_test]
        models[i]['K'] = K
        models[i]['arch'] = hiddenSize
    
    return models

data = load_iris()
X , y = data.data , data.target
safety = 10
modelsWanted = 10 + safety
models = run(modelsWanted , X , y)

def modelsInfo():
    return models
