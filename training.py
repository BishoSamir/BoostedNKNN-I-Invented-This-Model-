import torch 
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from data import getData
from neural import ANN

models , yTrue = getData() 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nModels = len(models.keys())
epochs = 200
modelsWanted = 10 # has to be less than nModels 

def getCurrentModel(index):
    output_size = 3       # number of categories in target
    train_loader , test_loader , totalInputSize , secondInputSize ,hidden_size  = models[index]
    model = ANN(totalInputSize-secondInputSize , secondInputSize  , output_size , hidden_size)
    optimizer = optim.Adam(model.parameters() , lr = 3e-4 , weight_decay=3e-3)
    criterion = nn.CrossEntropyLoss(reduction = 'none')

    return train_loader , test_loader , model , optimizer , criterion 


def trainingStep(model , train_loader , criterion , optimizer , device ,keepLoss , prevLoss):
    
    model.train()
    train_loss , correct , samples = 0 , 0 , 0
    keepLosses = {}
    for batchIdx , (x , y) in enumerate(train_loader):
        x , y = x.to(device) , y.to(device)
        yPred = model(x)
        
        if prevLoss.keys():
            loss = criterion(yPred , y) * prevLoss[batchIdx]
        else :
            loss = criterion(yPred , y)
        
        if keepLoss :
            keepLosses[batchIdx] = loss
        
        loss = torch.mean(loss)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct += (y == yPred.argmax(dim = 1)).sum()
        samples += len(y)
        
    train_loss /= len(train_loader)
    acc = round(float(correct/samples)*100,3)
    
    return train_loss , acc , keepLosses

def testingStep(model , test_loader , criterion , device , storePred):
    
    test_loss , correct , samples = 0 , 0 , 0
    model.eval()
    predLabels = []
    with torch.inference_mode():
        
        for x , y in test_loader:

            x , y = x.to(device) , y.to(device)
            yPred = model(x)
            if storePred:
                predLabels += yPred.argmax(dim = 1).tolist()

            loss = criterion(yPred , y)
            loss = torch.mean(loss)
            test_loss += loss.item()
            
            correct += (y==yPred.argmax(dim = 1)).sum()
            samples += len(y)
    
    test_loss /= len(test_loader)
    acc = round(float(correct/samples)*100,3)
    
    return test_loss , acc , predLabels

def removeGrad(dic):
    for key in dic.keys():
        dic[key] = dic[key].clone().detach()

def isOverfit(trainAcc , testAcc , threshold):
    return trainAcc - testAcc > threshold

def updateLoss(currLoss  , idx):
    if(origLoss.keys()):
        for key in prevLoss.keys():
            origLoss[key] += currLoss[key]
            prevLoss[key] =  origLoss[key]  / (idx+1)
    else:
        for key in currLoss.keys():
            origLoss[key] = prevLoss[key] = currLoss[key]

def decreaseAttention(acc):
    arr = np.array([])
    for key in prevLoss.keys():
        arr = np.append(arr , prevLoss[key].cpu().numpy())
    
    new_Weights = getOutliers(arr , acc)
    prevLen = 0
    for key in prevLoss.keys():
        currLen = len(prevLoss[key])
        prevLoss[key] = prevLoss[key] * new_Weights[prevLen : prevLen + currLen]
        prevLen += currLen

def getOutliers(arr ,acc):
    z_score = (arr - np.mean(arr) ) / np.std(arr)
    new_Weights = np.where(z_score > 2 , np.exp(-acc/100) , 1)
    return new_Weights

All_train_loss_list = []
All_val_loss_list = []
prevLoss = {}
origLoss = {}
modelsInventory = {}
modelID = 0

for idx in range(nModels):
    train_loader , test_loader , model , optimizer , criterion = getCurrentModel(idx)
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        train_loss , train_acc , currLoss = trainingStep(model , train_loader , criterion , optimizer ,
                                                        device , epoch + 1 == epochs , prevLoss)
        test_loss , test_acc , predLabels = testingStep(model , test_loader , criterion , device , epoch + 1 == epochs)
        
        train_loss_list.append(train_loss)
        val_loss_list.append(test_loss)
        
        if(epoch + 1 == epochs):

            if( not isOverfit(train_acc , test_acc , 7) ):
                removeGrad(currLoss)
                updateLoss(currLoss , idx)

            else:
                
                if( not prevLoss.keys()):
                    break 
                decreaseAttention(test_acc)
                break
            
            modelsInventory[modelID] = (model , predLabels ,  test_acc/100 )
            modelID += 1
            print(f" train_loss= {train_loss} |train_acc = {train_acc}| test_loss= {test_loss}| accuracy= {test_acc}")

    All_train_loss_list.append(train_loss_list)
    All_val_loss_list.append(val_loss_list)
    if(modelID == modelsWanted):
        break

def modelsWithPredictions():
    return modelsInventory , yTrue