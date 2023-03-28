import numpy as np

class preNeural:

    def __init__(self, k):
        self.k = k

    def getRightData(self, X , y , isTrain = 1):
        self.X = X 
        self.y = y
        self.observations , self.features = X.shape
        return self.preprocess(X , isTrain)

    
    def preprocess(self, X , isTrain):
        predictions = [np.append(self._preprocess(x , isTrain) ,x).reshape(-1) for x  in X ]
        return predictions
    
    
    def _preprocess(self, x , isTrain):
        if(self.features / self.observations >=1):
            distances = np.sum( abs(self.X - x) , axis = 1)
        else:
            distances = np.sum( (self.X - x)**2  , axis = 1)**0.5 
        
        totalInfo = []
        
        for i in range( self.observations ):
            arr = np.array([distances[i] , self.y[i]])
            arr = np.append(arr , self.X[i])          
            totalInfo.append(arr)
        
        totalInfo = sorted(totalInfo , key = lambda x: x[0])
        
        if(isTrain):
            return np.array(totalInfo[1 : self.k + 1 ]).reshape(-1)
        
        return np.array(totalInfo[:self.k]).reshape(-1)
        
    
