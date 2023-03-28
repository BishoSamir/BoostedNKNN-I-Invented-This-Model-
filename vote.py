from training import modelsWithPredictions
import pandas as pd 
import numpy as np
import torch 
import torch.nn.functional as F

models , yTrue = modelsWithPredictions()
allPred = []
names = []

def amountOfSay():
    acc = []
    idx = 1
    for key in models.keys():
        acc.append(models[key][-1])
        allPred.append(models[key][1])
        names.append(f'clf{idx}')
        idx += 1
    
    acc = torch.tensor(acc , dtype = torch.float32)
    return F.softmax(acc , dim = 0)

def vote(row , voteWeight):
    dic = {}
    for i in row:
        dic[i] = 0
    
    maxKey = None
    cumWeight = 0

    for idx , pred in enumerate(row):
        dic[pred] += voteWeight[idx]
        if(dic[pred] > cumWeight):
            cumWeight = dic[pred]
            maxKey = pred
    
    return maxKey    


voteWeight = amountOfSay().cpu().numpy()
df = pd.DataFrame(np.array(allPred).T , columns = names)

df['finalVote'] = df.apply(lambda x: vote( [x[i] for i in df.columns] , voteWeight) , axis = 1)
df['y_true'] = yTrue

print(f'final accuracy = { np.sum(df["finalVote"] == df["y_true"]) / df.shape[0] }')