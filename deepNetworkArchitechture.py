import torch
import itertools as it
import torch.nn as nn


class ConvNet(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w = tuple(self.in_size)  # C,H,W

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        '''
        N is the total number of convolutional layers,
        P specifies how many convolutions to perform before each pooling layer

        Note: 
        every convolution layer - could technically reduce the image size - but it will not because of the parameters we chose for Conv2d.
        FURTHERMORE: the number of channels in each layer, is a function ONLY of the number of filters (= the number of kernels)  

        '''
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        self.num_pools_performed = 0

        curr_channels = in_channels
        for i in range(N):
            layers.append(
                nn.Conv2d(in_channels=curr_channels, out_channels=self.channels[i], kernel_size=3, padding=1, stride=1,
                          dilation=1))
            curr_channels = self.channels[i]
            layers.append(nn.ReLU())
            # if ((i + 1) % P) == 0:
            #     layers.append(nn.MaxPool2d(kernel_size=2))
            #     self.num_pools_performed += 1

        # ========================

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.

        # ====== YOUR CODE: ======

        '''
        Note: 
        every convolution layer - could technically reduce the image size - but it will not because of the parameters we chose for Conv2d.
        FURTHERMORE: the number of channels in each layer, is a function ONLY of the number of filters (= the number of kernels)  
        '''
        '''
        we want to calculate the starting_num_features to input to our classifier sequential.
        in order to do that, we need to calculate the size of last output from the convolution layers
        that can be achieved by finding out how many pools were performed - because for every pool, the original image's size
        was reduced by /2.

        then, to calc. starting_num_features, we need to get the final C x H x W from the convolutional layers.
        and  so:

                                                    C                   x   H     x   W

        starting_num_features =    conv layers final num of channells      in_h      in_w
                                    _________________________________   * ______  *________
                                                    1                     factor    factor    
        '''

        M = len(self.hidden_dims)

        size_reduction_factor = 2 ** self.num_pools_performed
        starting_num_features = self.channels[-1] * (in_h // size_reduction_factor) * (in_w // size_reduction_factor)

        curr_channels = starting_num_features
        for i in range(M):
            layers.append(nn.Linear(in_features=curr_channels, out_features=self.hidden_dims[i]))
            layers.append(nn.ReLU())
            curr_channels = self.hidden_dims[i]

        layers.append(nn.Linear(in_features=self.hidden_dims[M - 1], out_features=self.out_classes))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        # features_seq = self.feature_extractor()
        # classifier_seq = self.classifier()

        '''
        ## input image: 1,3,32,32   ->>> after convolutions: 1,32,new,new
        ## features dimensions are: (num of inputs, num of channels, hight, width)  == 4 dimensions
        ## classifier needs 2 dimensions (num of inputs, num of feature)  == 2 dimensions
        ## so we need to transform:  (num of inputs, num of channels*hight*width)
         ## features_flattened size is 1,20000
        '''
        features = self.feature_extractor(input=x)
        features_flattened = features.view(features.size(0), -1)  # -1 means inferring from other dimensions
        out = self.classifier(features_flattened)  # out is the class scores

        # ========================
        return out