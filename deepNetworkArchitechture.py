import torch
import itertools as it
import torch.nn as nn


class ConvNet(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list, pool_every: int, hidden_dims: list):
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
        self.predictor = self._make_predictor()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w = tuple(self.in_size)  # C,H,W

        layers = []

        '''
        Structure:
        if pools are wanted:
                [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        or if no pools wanted:
                (CONV -> ReLU)*N

        N is the total number of convolutional layers,
        P specifies how many convolutions to perform before each pooling layer

        Note: 
        every convolution layer - could technically reduce the image size - but it will not because of the parameters we chose for Conv2d.
        FURTHERMORE: the number of channels in each layer, is a function ONLY of the number of filters (= the number of kernels)  

        '''
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


        seq = nn.Sequential(*layers)
        return seq

    def _make_predictor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []

        '''
        Structure:
                (Linear -> ReLU)*M -> Linear
        
        #  we first need to calculate the number of features going in to the first linear layer.
        #  The last Linear layer has an output dim of out_classes.

        Note: 
        every convolution layer - could technically reduce the image size - but it will not because of the parameters we chose for Conv2d.
        FURTHERMORE: the number of channels in each layer, is a function ONLY of the number of filters (= the number of kernels)  
        
        Note:
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
        # 
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):

        features = self.feature_extractor(input=x)
        features_flattened = features.view(features.size(0), -1)  # -1 means inferring from other dimensions
        out = self.predictor(features_flattened)  

        return out


class EncoderFullyConnected(nn.Module): 
    def __init__(self, in_features, connected_layers_dim_list ,out_features):
        super().__init__()

        # save data
        self.in_features = in_features
        self.connected_layers_dim_list = connected_layers_dim_list
        self.out_features = out_features
        self.num_of_hidden_dims = len(connected_layers_dim_list)
        # convinience
        N = self.num_of_hidden_dims
        # init the layer list
        layers = []
        # add all layers but the last one
        current_num_of_features = in_features
        for i in range(self.num_of_hidden_dims):
            layers.append(nn.Linear(in_features=current_num_of_features, out_features=connected_layers_dim_list[i]))
            layers.append(nn.ReLU())
            current_num_of_features = connected_layers_dim_list[i]
        # add the last layer (outside of the loop)
        layers.append(nn.Linear(in_features=connected_layers_dim_list[N-1], out_features=out_features))

        # create the network
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class DecoderFullyConnected(nn.Module): 
    def __init__(self, in_features, connected_layers_dim_list ,out_features):
        super().__init__()

        # save data
        self.in_features = in_features
        self.connected_layers_dim_list = connected_layers_dim_list
        self.out_features = out_features
        self.num_of_hidden_dims = len(connected_layers_dim_list)
        # convinience
        N = self.num_of_hidden_dims
        # init the layer list
        layers = []
        # add all layers but the last one
        current_num_of_features = in_features
        for i in range(self.num_of_hidden_dims):
            layers.append(nn.Linear(in_features=current_num_of_features, out_features=connected_layers_dim_list[i]))
            layers.append(nn.ReLU())
            current_num_of_features = connected_layers_dim_list[i]
        # add the last layer (outside of the loop)
        layers.append(nn.Linear(in_features=connected_layers_dim_list[N-1], out_features=out_features))

        # create the network
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class AutoencoderNet(nn.Module):
    def __init__(self, in_features, connected_layers_dim_list, z_dim, batch_size, device):
        """
        AutoencoderNet initialization
        """
        super().__init__()

        # anouncement
        print(f'\nentered __init__ of AutoencoderNet')

        # save data from outside
        self.device = device
        self.batch_size = batch_size
        self.in_features = in_features # note: this number is NOT batch size dependant ! will be handeled soon
        self.z_dim = z_dim             # note: this number is NOT batch size dependant ! will be handeled soon

        # handle the effect of batch size on the number of features
        num_of_in_features = batch_size * in_features  #NOTE: this is because in the network I am flattenning the tensor
        num_of_out_features = batch_size * z_dim

        # small prep
        reversed_list = list(reversed(connected_layers_dim_list))

        # create the decoder and the encoder
        self.encode = EncoderFullyConnected(in_features=num_of_in_features, connected_layers_dim_list=connected_layers_dim_list, out_features=num_of_out_features)  
        self.decode = DecoderFullyConnected(in_features=num_of_out_features, connected_layers_dim_list=reversed_list, out_features=num_of_in_features)  
        

    def encodeWrapper(self, x):
        '''
        #  Convert an input vector x to a  latent vector z as output
        '''
        ## flatten the input to enter the model 
        x_orig_shape = x.shape
        x_flattenned = x.flatten()
        ## get the encoder output
        encoder_output = self.encode(x_flattenned)
        ## set z to be the output but after changing the dimensions correctly
        z = encoder_output.view(self.batch_size, self.z_dim)  

        return z

    
    def decodeWrapper(self, z):
        '''
        #  Convert a latent vector z back into a reconstructed output x_reconstructed.
        '''
        ## flatten the input to enter the model 
        z_orig_shape = z.shape
        z_flattenned = z.flatten()

        ## get the encoder output
        decoder_output = self.decode(z_flattenned)
        ## set z to be the output but after changing the dimensions correctly
        x_reconstructed = decoder_output.view(self.batch_size, self.in_features) 

        return x_reconstructed


    def forward(self, x):
        z = self.encodeWrapper(x)
        return self.decodeWrapper(z)