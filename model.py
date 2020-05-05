''' Neural Network class

Hidden Layers
===============================
    1. fully-connected   | ReL
    2. fully-connected   | ReL
    3. fully-connected   | ReL
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    '''Model'''
    
    def __init__(self, state_size, action_size, seed, 
                 fc1_units=128, fc2_units=64, fc3_units=32):
        '''Init parameters
        Params
        ======
            state_size (int): dimension of state (input)
            action_size (int): dimension of action (output)
            seed (int): random seed
        '''
        super(QNetwork, self).__init__() # (execute parent's __init__)
        self.seed = torch.manual_seed(seed)
        
        # define dims of hidden layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size) # output
       
    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x