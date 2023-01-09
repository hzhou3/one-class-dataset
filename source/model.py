__author__ = 'Hao - 6/2020'

import torch
import torchvision
from torch import nn
import torch.nn.functional as F


'''
model instance, code size is 90. 
Can be changed to any number. e.g., 32
but in uti.py, line 230-line 232 needs to be changed to the same dimension.
'''
class autoencoder(nn.Module):
    def __init__(self, dim_in, gabor=False):
        super(autoencoder, self).__init__()
        self.dim_in  = dim_in
        self.dim_out = dim_in
        ###################################### #####################
        if gabor:
            self.encoder = nn.Sequential(
                # self.fc1
                # self.h1
                # nn.ReLU(),
                # nn.Dropout(0.05),
                
                nn.Linear(in_features=self.dim_in, out_features=90),
                # nn.BatchNorm1d(32),
                # nn.Sigmoid(),
                # nn.Linear(in_features=128, out_features=64),
                # nn.Linear(in_features=64, out_features=32)

                )
            self.decoder = nn.Sequential(
                # self.h2,
                # self.fc2
                # nn.ReLU(),

                # nn.Linear(in_features=32, out_features=64),
                # nn.Linear(in_features=64, out_features=128),
                nn.Linear(in_features=90, out_features=self.dim_out),
                # nn.BatchNorm1d(180),
                # nn.Sigmoid(),
                )
        else:
            # this is 1D convolution
            self.encoder = nn.Sequential(
                    nn.Conv1d(2, 16, 45, stride = 45),
                    nn.ReLU(),
                    # nn.Conv1d(4, 8, 5, stride = 3),
                    # nn.ReLU(),
                    # nn.MaxPool1d(3, stride=1)
                ) 
            self.decoder = nn.Sequential(
                    # nn.ConvTranspose1d(8, 4, 5, stride=3),  
                    # nn.ReLU(),
                    nn.ConvTranspose1d(16, 2, 45, stride=45),
                    nn.ReLU()
                )
       

    def forward(self, x):
        # x = x.view(-1, 2, 90)
        code = self.encoder(x)
        x = self.decoder(code)
        # return x, code
        return x, code#x.view(-1, 180)
