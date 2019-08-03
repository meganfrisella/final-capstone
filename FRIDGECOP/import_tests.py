import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from update_fridge import remove_item, layer_image, propose_regions, parse_food
import math
from collections import defaultdict
import pickle

from torch import nn
import torch.nn.functional as F

from detection_utils.boxes import non_max_suppression, generate_targets
from detection_utils.metrics import compute_recall, compute_precision
from detection_utils.pytorch import softmax_focal_loss
from mynn.layers.conv import conv
from mynn.layers.dense import dense
from mynn.layers.dropout import dropout
from mynn.activations.relu import relu
from mynn.initializers.glorot_uniform import glorot_uniform
from mygrad.nnet.layers import max_pool
from mygrad.nnet.losses import softmax_crossentropy
from mygrad.nnet.layers import conv_nd
from mynn.losses.cross_entropy import softmax_cross_entropy
from mynn.optimizers.adam import Adam

num_categories = 2
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #Check this: the 3s-- kernel size
        
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 30, 3)
        self.conv4 = nn.Conv2d(30, 40, (59,35))
        
        self.change = num_categories #used to be 4
        self.classification = nn.Conv2d(40, self.change, 1) # background / rectangle / triangle / circle
        self.regression = nn.Conv2d(40, 4, 1)
        
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4,
                     self.classification, self.regression):
            nn.init.xavier_normal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.classification.bias[0], -4.6)  # rougly -log((1-π)/π) for π = 0.01
        
    def forward(self, x):
        
        #print("X")
        #print(x.shape)
        #print("Here3", F.max_pool2d(F.relu(self.conv1(x)), 2))
        #print()
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        #print(x.shape)
        
        #Check this:
        #print("x", x.shape)
        #print(self.classification(x).dtype)
        classifications = self.classification(x).permute(0, 2, 3, 1)                          # (N, R, C, # classes)
        classifications = classifications.reshape(x.shape[0], -1, classifications.shape[-1])  # (N, R*C, # classes)
        regressions = self.regression(x).permute(0, 2, 3, 1)                                  # (N, R, C, # classes)
        regressions = regressions.reshape(x.shape[0], -1, 4)                                  # (N, R*C, 4)
        return classifications, regressions
#CUT AND PASTE [1] END

#Check this: temporary change
#CUT AND PASTE [0] START
class ClassifyingModel():
    def __init__(self):
        #Check this: the 3s-- kernel size
        gain = {'gain': np.sqrt(2)}
        
        self.conv1 = conv(3, 20, (5,5), weight_initializer=glorot_uniform, 
      weight_kwargs=gain)
        self.conv2 = conv(20, 10, (5,5), weight_initializer=glorot_uniform, 
      weight_kwargs=gain)
        
        #Check the dimensions on this:
        self.dense3 = dense(49000 , 232, weight_initializer=glorot_uniform, 
      weight_kwargs=gain)
        
    def __call__(self, x):
        ''' Forward data through the network.
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, D)
            The data to forward through the network.
            
        Returns
        -------
        mygrad.Tensor, shape=(N, 1)
            The model outputs.
        '''
        # returns output of dense -> relu -> dense -> relu -> dense -> softmax three layer.
        #print(x.shape)
        a = self.conv1(x)
        #print(a.shape)
        l = relu(max_pool(a,pool=(2,2), stride=1))
        #print(l.shape)
        b = max_pool(self.conv2(l), pool=(2,2),stride=1)
        #print(b.shape)
        s = b.shape
        #print(temp1.shape)
        b = b.reshape(s[0], s[1]*s[2]*s[3])
        #print(b.shape)
        #print(temp1.shape)
        return self.dense3(relu(b))
    
    @property
    def parameters(self):
        ''' A convenience function for getting all the parameters of our model. '''
        return self.conv1.parameters + self.conv2.parameters + self.dense3.parameters
#CUT AND PASTE [0] END