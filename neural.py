import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self ,first_input , second_input , output_size , hidden_size ):
        super().__init__()
        
        self.first_input = first_input
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(first_input , hidden_size) 
        self.fc2 = nn.Linear(second_input , hidden_size)
        self.relu = nn.ReLU()
        self.lastLayer = nn.Linear(hidden_size * 2 , output_size)

    def forward(self , x ):
        x1 , x2 = x[: , :self.first_input], x[: , self.first_input:]
        x1 = self.relu(self.fc1(x1))
        x2 = self.relu(self.fc2(x2))
        x = torch.cat((x1,x2) , 1)
        return self.lastLayer(x)
        
        


    