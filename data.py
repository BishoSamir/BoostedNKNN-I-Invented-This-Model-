from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader 
import numpy as np 
import torch
from preNeural import preNeural
from manage import modelsInfo

modelsDetails = modelsInfo()

def dataPreprocessing(train_data, test_data , y_train , y_test):
    X_tensor = torch.tensor(train_data , dtype = torch.float32)
    y_tensor = torch.tensor(y_train , dtype = torch.long)
    df_train = TensorDataset(X_tensor , y_tensor)
    
    X_tensorTest = torch.tensor(test_data , dtype = torch.float)
    y_tensorTest = torch.tensor(y_test , dtype = torch.long)
    df_test = TensorDataset(X_tensorTest , y_tensorTest)

    train_loader = DataLoader(df_train , batch_size=16 , shuffle = True)
    test_loader = DataLoader(df_test , batch_size=16 , shuffle = False )
    
    return train_loader , test_loader

def getData():
    modelsInventory = {}
    for i in range( len( modelsDetails.keys() ) ):
        X_train , X_test , y_train , y_test = modelsDetails[i]['data']
        
        preModel = preNeural(modelsDetails[i]['K'])
        train_data = np.matrix( preModel.getRightData(X_train , y_train , 1) )
        test_data = np.matrix( preModel.getRightData(X_test , y_test , 0) )
        
        train_loader , test_loader = dataPreprocessing(train_data, test_data , y_train , y_test)
        modelsInventory[i] = [ train_loader , test_loader  , train_data.shape[1]  
                        ,  X_train.shape[1] , modelsDetails[i]['arch']  ]
    
    return modelsInventory , y_test

