import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image
%matplotlib notebook
from pathlib import Path
from noggin import create_plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
%matplotlib notebook
import pickle
import torch

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

import numpy as np
import mygrad as mg
from mygrad import Tensor

from noggin import create_plot
import matplotlib.pyplot as plt

%matplotlib notebook

def compute_detections(classifications, regressions, feature_map_width, anchor_box_step, anchor_box_size):
    """ Compute a set of boxes, class predictions, and foreground scores from
        detection model outputs.

    Parameters
    ----------
    classifications : torch.Tensor, shape=(N, R*C, # classes)
        A set of class predictions at each spatial location.

    regressions : torch.Tensor, shape=(N, R*C, 4)
        A set of predicted box offsets, in (x, y, w, h) at each spatial location.

    feature_map_width : int
        The number of pixels in the feature map, along the x direction.

    anchor_box_step : int
        The number of pixels (in image space) between each anchor box.

    anchor_box_size : int
        The side length of the anchor box.

    Returns
    -------
    Tuple[numpy.ndarray shape=(R*C, 4), numpy.ndarray shape=(R*C, 1), numpy.ndarray shape=(R*C,)]
        The (boxes, class predictions, foreground scores) at each spatial location.
    """
    
    #print(regressions)
    #print(regressions.shape)
    #print(len(regressions))
    
    box_predictions = np.empty((len(regressions), 4), dtype=np.float32)
    scores = torch.softmax(classifications, dim=-1).detach().cpu().numpy()
    scores = 1 - scores[:, 0]  # foreground score

    class_predictions = classifications.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    regressions = regressions.detach().cpu().numpy()

    y, x = np.divmod(np.arange(len(classifications)), feature_map_width, dtype=np.float32)
    x_reg, y_reg, w_reg, h_reg = regressions.T  # transform (R*C, 4) to (4, R*C) for assignment
    x = anchor_box_step * x + anchor_box_size * x_reg
    y = anchor_box_step * y + anchor_box_size * y_reg

    half_w = np.clip(np.exp(w_reg), 0, 10**6) * anchor_box_size / 2
    half_h = np.clip(np.exp(h_reg), 0, 10**6) * anchor_box_size / 2

    box_predictions[:, 0] = x - half_w  # x1
    box_predictions[:, 1] = y - half_h  # y1
    box_predictions[:, 2] = x + half_w  # x2
    box_predictions[:, 3] = y + half_h  # y2

    return box_predictions, class_predictions, scores


#Check this: temporary change
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

        nn.init.constant_(self.classification.bias[0], -4.6)  # rougly -log((1-¦Ð)/¦Ð) for ¦Ð = 0.01
        
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


#Check this function
def crop_image(image, left, top):
    #assumes the dimensions of the resulting image should be 80X80
    return image[:, :, left:left+80, top:top+80]

#last known mean and std values: 200.62128, 42.149155 respectively

def scan_fridge(fridge_image, training_mean, training_std):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    f = open("Actual_Model.p","rb")
    model = pickle.load(f)
    f.close()
    img = np.copy(fridge_image).astype(np.float32)
    img-=training_mean
    img/= training_std
    img = img[np.newaxis]
    img = torch.tensor(img.transpose(0, 3, 1, 2)).to(device)
    
    assert img.shape == torch.Size([1, 3,  800, 497]), "The fridge image is not in the right shape"
    
    out_cls, out_reg = model(img)
    box_preds, class_preds, scores = compute_detections(out_cls.squeeze(), 
                                                        out_reg.squeeze(), 
                                                        feature_map_width=13,
                                                        anchor_box_step=40, 
                                                        anchor_box_size=80)

    keep = scores > threshold
    box_preds = box_preds[keep]
    class_preds = class_preds[keep]
    scores = scores[keep]
    keep_idxs = non_max_suppression(box_preds, scores, threshold=0.1)
    box_preds = box_preds[keep_idxs]
    class_preds = class_preds[keep_idxs]
    
    food_images = []
    food_coords = []
    
    
    for class_pred, box_pred in zip(class_preds, box_preds):
        if class_pred > 0:
            #print("Here")
            x1, y1, x2, y2 = box_pred
            food_images.append(crop_image(fridge_image,x1,y1))
            food_cords.append((x1,y1))
    food_images = np.array(food_images)
    f = open("classifying_model1", "rb")
    classifying_model = pickle.load(f)
    f.close()
    classes = classifying_model(food_images)
    return [Item(food_coords[i][0], food_coords[i][1], labels[classes[i]], "food")]


#Check this: temporary change
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
        temp1 = max_pool(
                self.conv2(relu(max_pool(self.conv1(x), pool=(2,2),stride=1))), 
                pool=(2,2),stride=1)
        s = temp1.shape
        #print(temp1.shape)
        temp1 = temp1.reshape(s[0], s[1]*s[2]*s[3])
        #print(temp1.shape)
        return self.dense3(relu(temp1))
    
    @property
    def parameters(self):
        ''' A convenience function for getting all the parameters of our model. '''
        return self.conv1.parameters + self.conv2.parameters + self.dense3.parameters


rgba_image = mpimg.imread('food.png')

rgba_pil = PIL.Image.open('food.png')
img = np.array(rgba_pil.convert('RGB'))

row_step = 1200//15
col_step= 1280//16
shelf_coord = [260, 380, 500, 620, 770] #coordinates of the first, second ... shelves


#plt.imshow(img)

images = []
labels = np.arange(232)
for i in range(0, 1200, row_step):
    for j in range(0, 1280, col_step):
        images.append(img[i: i + row_step, j : j + col_step, :])
images = images[:-8]

x_train = np.array(images)
y_train = labels

x_test = []
y_test = []

#Check this: change to 500 later
for i in range(50):
    ind = np.random.randint(0,232)
    x_test.append(x_train[ind])
    y_test.append(ind)
    

x_train = x_train.swapaxes(1,3).swapaxes(2,3) 
x_test = np.array(x_test).swapaxes(1,3).swapaxes(2,3)
y_test = np.array(y_test)


def accuracy(predictions, truth):
    """
    Returns the mean classification accuracy for a batch of predictions.
    
    Parameters
    ----------
    predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
        The scores for D classes, for a batch of M data points
    truth : numpy.ndarray, shape=(M,)
        The true labels for each datum in the batch: each label is an
        integer in [0, D)
    
    Returns
    -------
    float
    """
    if isinstance(predictions, mg.Tensor):
        predictions = predictions.data
    return np.mean(np.argmax(predictions, axis=1) == truth)

import numpy as np
# Set `batch_size = 100`: the number of predictions that we will make in each training step

# <COGINST>
batch_size = 100
# </COGINST>

# We will train for 10 epochs; you can change this if you'd like.
# You will likely want to train for much longer than this
for epoch_cnt in range(15):
    
    # Create the indices to index into each image of your training data
    # e.g. `array([0, 1, ..., 9999])`, and then shuffle those indices.
    # We will use this to draw random batches of data
    # <COGINST>
    idxs = np.arange(len(x_train))  # -> array([0, 1, ..., 9999])
    np.random.shuffle(idxs)  
    # </COGINST>
    
    for batch_cnt in range(0, len(x_train) // batch_size):
        # Index into `x_train` to get your batch of M images.
        # Make sure that this is a randomly-sampled batch
        # <COGINST>
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_train[batch_indices]  # random batch of our training data
        # </COGINST>
        
        # compute the predictions for this batch by calling on model
        # <COGINST>
        prediction = classifying_model(batch)
        # </COGINST>

        # compute the true (a.k.a desired) values for this batch: 
        # <COGINST>
        truth = y_train[batch_indices]
        # </COGINST>

        # compute the loss associated with our predictions(use softmax_cross_entropy)
        # <COGINST>
        loss = softmax_cross_entropy(prediction, truth)
        # </COGINST>

        # compute the accuracy between the prediction and the truth 
        # <COGINST>
        acc = accuracy(prediction, truth)
        # </COGINST>

        # back-propagate through your computational graph through your loss
        # <COGINST>
        loss.backward()
        # </COGINST>

        # execute gradient-descent by calling step() of optim
        # <COGINST>
        optim.step()
        # </COGINST>
        
        # null your gradients
        # <COGINST>
        loss.null_gradients()
        # <COGINST>

        plotter.set_train_batch({"loss" : loss.item(),
                                 "accuracy" : acc},
                                 batch_size=batch_size)
    
    # After each epoch we will evaluate how well our model is performing
    # on data from cifar10 *that it has never "seen" before*. This is our
    # "test" data. The measured accuracy of our model here is our best 
    # estimate for how our model will perform in the real world 
    # (on 32x32 RGB images of things in this class)
    test_idxs = np.arange(len(x_test))
    
    for batch_cnt in range(0, len(x_test)//batch_size):
        batch_indices = test_idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        
        batch = x_test[batch_indices]
        truth = y_test[batch_indices]
        
        # Get your model's predictions for this test-batch
        # and measure the test-accuracy for this test-batch
        # <COGINST>
        prediction = model(batch)
        test_accuracy = accuracy(prediction, truth)
        # </COGINST>
        
        # pass your test-accuracy here; we used the name `test_accuracy`
        plotter.set_test_batch({"accuracy" : test_accuracy}, batch_size=batch_size)
    plotter.set_test_epoch()

    
