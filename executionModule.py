import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
from sklearn.decomposition import NMF
from deepNetworkArchitechture import ConvNet


def train_prediction_model(model_to_train, ds_train, dl_train, loss_fn, optimizer, num_of_epochs_wanted, max_alowed_number_of_batches, device):
    '''
    preparations
    '''
    print("/ * \ ENTERED train_prediction_model / * \ ")
    # for me
    num_of_epochs = num_of_epochs_wanted
    model = model_to_train
    # important: load model to cuda
    if device.type == 'cuda':
        model = model.to(device=device) 
    # print info for user
    print(f'recieved model {model}\nrecieved loss_fn {loss_fn}\nrecieved optimizer {optimizer}\nrecieved num_of_epochs_wanted {num_of_epochs_wanted}\nrecieved max_alowed_number_of_batches {max_alowed_number_of_batches}')

    # compute actual number of batches to train on in each epoch
    num_of_batches = (len(ds_train) // dl_train.batch_size)  # TODO: check this line
    if num_of_batches > max_alowed_number_of_batches:
        print(f'NOTE: in order to speed up training (while damaging accuracy) the number of batches per epoch was reduced from {num_of_batches} to {max_alowed_number_of_batches}')
        num_of_batches = max_alowed_number_of_batches


    '''
    BEGIN TRAINING !!!
    # note 2 loops here: external (epochs) and internal (batches)
    '''    
    print("****** begin training ******")

    for iteration in range(num_of_epochs):
        print(f'\niteration {iteration+1} of {num_of_epochs} epochs')
        
        # init variables for external loop
        dl_iter = iter(dl_train)  # iterator over the dataloader. called only once, outside of the loop, and from then on we use next() on that iterator
        loss_values_list = []
        num_of_correct_predictions_this_epoch = 0

        for batch_index in range(num_of_batches):
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            # get current batch data 
            data = next(dl_iter)  # note: "data" variable is a list with 2 elements:  data[0] is: <class 'torch.Tensor'> data[1] is: <class 'torch.Tensor'>
            x, y = data  # note :  x.shape is: torch.Size([25, 3, 176, 176]) y.shape is: torch.Size([25]) because the batch size is 25
            
            # TODO: check if .to(device=device) is needed in both vars (030920 test)
            if device.type == 'cuda':
                x = x.to(device=device)  
                y = y.float()
                y = y.to(device=device)
            
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)  
            
            # TODO: check if .to(device=device) is needed in both vars (030920 test)
            if device.type == 'cuda':
                y_pred = y_pred.float()
                y_pred = y_pred.squeeze()
                y_pred = y_pred.to(device=device)

            # checking for same values in the predcition y_pred as in ground truth y
            y_pred_rounded = torch.round(y_pred).type(torch.int).flatten()  #TODO: note this rounding process to the closest iteger !
            num_of_correct_predictions_this_batch = torch.eq(y, y_pred_rounded).sum()
            num_of_correct_predictions_this_epoch += num_of_correct_predictions_this_batch

            # Compute (and save) loss.
            loss = loss_fn(y_pred, y)  # todo: check order
            loss_values_list.append(loss.item())

            # Before the backward pass, use the optimizer object to zero all of the gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are accumulated in buffers( i.e, not overwritten) 
            # whenever ".backward()" is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        #end of inner loop
        print(f'\nfinished inner loop.\n')


        # data prints on the epoch that ended
        print(f'in this epoch: min loss {np.min(loss_values_list)} max loss {np.max(loss_values_list)}')
        print(f'               average loss {np.mean(loss_values_list)}')
        print(f'               number of correct predictions: {num_of_correct_predictions_this_epoch} / {dl_train.batch_size*num_of_batches}')

 
    print(f'finished all epochs !')
    print(" \ * / FINISHED train_prediction_model \ * / ")
    return model


def runExperimentWithModel_BasicConvNet(dataset : Dataset, hyperparams, device):
    '''
    hyperparams is a dict that should hold the following variables:
    batch_size, max_alowed_number_of_batches, precent_of_dataset_allocated_for_training, learning_rate, num_of_epochs,   
    channels, num_of_convolution_layers, hidden_dims, num_of_hidden_layers, pool_every
    '''
    ###
    '''
    prep our dataset and dataloaders
    '''
    train_ds_size = int(len(dataset) * hyperparams['precent_of_dataset_allocated_for_training'])
    test_ds_size = len(dataset) - train_ds_size
    split_lengths = [train_ds_size, test_ds_size]  
    ds_train, ds_test = random_split(dataset, split_lengths)
    dl_train = DataLoader(ds_train, hyperparams['batch_size'], shuffle=True)
    dl_test = DataLoader(ds_test, hyperparams['batch_size'], shuffle=True)

    '''
    prepare model, loss and optimizer instances
    '''
    x0, y0 = dataset[0]
    in_size = x0.shape # note: if we need for some reason to add batch dimension to the image (from [3,176,176] to [1,3,176,176]) use x0 = x0.unsqueeze(0)  # ".to(device)"
    output_size = 1 if isinstance(y0, int) else y0.shape # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)  !!! #TODO: might need to be changed to a single number using .item() ... or .squeeze().item()
    
    # create the model
    model = ConvNet(in_size, output_size, channels=hyperparams['channels'], pool_every=hyperparams['pool_every'], hidden_dims=hyperparams['hidden_dims'])
    # create the loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    '''
    now we can perform the test !
    '''

    model = train_prediction_model(model_to_train=model, ds_train=ds_train, dl_train=dl_train, loss_fn=loss_fn, 
                                   optimizer=optimizer, num_of_epochs_wanted=hyperparams['num_of_epochs'], 
                                   max_alowed_number_of_batches=hyperparams['max_alowed_number_of_batches'],
                                   device=device)

    # TODO: test the model ?
    pass


def runExperiment1_singleGenePrediction(dataset : Dataset, device):
    
    print("\n----- entered function runExperiment1_singleGenePrediction -----")
    '''
    prep our dataset and dataloaders
    '''
    batch_size = 25
    train_ds_size = int(len(dataset) * 0.8)
    test_ds_size = len(dataset) - train_ds_size
    split_lengths = [train_ds_size, test_ds_size]  
    ds_train, ds_test = random_split(dataset, split_lengths)
    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size, shuffle=True)

    '''
    prepare model, loss and optimizer instances
    '''
    x0, _ = dataset[0]
    in_size = x0.shape # note: if we need for some reason to add batch dimension (from [3,176,176] to [1,3,176,176]) use x0 = x0.unsqueeze(0)  # ".to(device)"
    output_size = 1  #notes on this line:
        # formerly known as - "out_classes". now, this will be the regression value FOR EACH SINGLE IMAGE (or so i think) #TODO: verify
        # TODO: note that i did not yet perform in softmax or any such thing
    channels = [32]  # these are the kernels if i remember correctly
    num_of_convolution_layers = len(channels)
    hidden_dims = [100]
    num_of_hidden_layers = len(hidden_dims)
    pool_every = 9999  # because of the parametes above, this practically means never ...

    # create the model
    model = ConvNet(in_size, output_size, channels=channels, pool_every=pool_every, hidden_dims=hidden_dims)
    if device.type == 'cuda':
        model = model.to(device=device)  # 030920 test: added cuda
    # create the loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 
    num_of_epochs = 5
    max_alowed_number_of_batches = 99999  # the purpose of this var is if i dont realy want all of the batches to be trained uppon ... 
    # and so if this number is higher than the real number of batches - it means i will use all of the batchs for my traiining process
    # note that there are currently (030920) 120 batches - 120 batches * 25 images in each batch = 3000 images in ds_train

    '''
    now we can perform the test !
    '''

    model = train_prediction_model(model_to_train=model, ds_train=ds_train, dl_train=dl_train, loss_fn=loss_fn, 
                                   optimizer=optimizer, num_of_epochs_wanted=num_of_epochs, max_alowed_number_of_batches=max_alowed_number_of_batches)

    # TODO: test the model ?

    print("\n----- finished function runExperiment1_singleGenePrediction -----\n")

    pass


def runExperiment2_allGenePrediction_dimReduction_KHighestVariances(dataset : Dataset, device):
    
    print("\n----- entered function runTest2_allGenePrediction_dimReduction_KHighestVariances -----")

    #TODO NOTE: IMPORTANT !!!! at the time being 15:27 in 180920 thisis just a copy of the function above, but with
    #               output_size = K !


    '''
    prep our dataset and dataloaders
    '''
    batch_size = 25
    train_ds_size = int(len(dataset) * 0.8)
    test_ds_size = len(dataset) - train_ds_size
    split_lengths = [train_ds_size, test_ds_size]  
    ds_train, ds_test = random_split(dataset, split_lengths)
    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size, shuffle=True)

    '''
    prepare model, loss and optimizer instances
    '''
    x0, y0 = dataset[0]
    in_size = x0.shape # note: if we need for some reason to add batch dimension to the image (from [3,176,176] to [1,3,176,176]) use x0 = x0.unsqueeze(0)  # ".to(device)"
    output_size = y0.shape # == K  !!! #TODO: might need to be changed to a single number using .item() ... or .squeeze().item()
    channels = [32]  # these are the kernels if i remember correctly
    num_of_convolution_layers = len(channels)
    hidden_dims = [100]
    num_of_hidden_layers = len(hidden_dims)
    pool_every = 9999  # because of the parametes above, this practically means never ...

    # create the model
    model = ConvNet(in_size, output_size, channels=channels, pool_every=pool_every, hidden_dims=hidden_dims)
    if device.type == 'cuda':
        model = model.to(device=device)  # 030920 test: added cuda
    # create the loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 
    num_of_epochs = 5
    max_alowed_number_of_batches = 99999  # the purpose of this var is if i dont realy want all of the batches to be trained uppon ... 
    # and so if this number is higher than the real number of batches - it means i will use all of the batchs for my traiining process
    # note that there are currently (030920) 120 batches - 120 batches * 25 images in each batch = 3000 images in ds_train

    '''
    now we can perform the test !
    '''

    model = train_prediction_model(model_to_train=model, ds_train=ds_train, dl_train=dl_train, loss_fn=loss_fn, 
                                   optimizer=optimizer, num_of_epochs_wanted=num_of_epochs, max_alowed_number_of_batches=max_alowed_number_of_batches)

    # TODO: test the model ?


    print("\n----- finished function runTest2_allGenePrediction_dimReduction_KHighestVariances -----\n")

    pass


def runExperiment3_allGenePrediction_dimReduction_NMF(dataset : Dataset, device):
    
    print("\n----- entered function runTest3_allGenePrediction_dimReduction_NMF -----")


    print("test printing the ds:")  
    print(f'W len {len(dataset.W)}')
    print(f'W type {type(dataset.W)}')
    print(f'W shape {dataset.W.shape}')
    print(f'H len {len(dataset.H)}')
    print(f'H len {type(dataset.H)}')
    print(f'H shape {dataset.H.shape}')
    print(f'W len {len(dataset.W)}')

    

    print("\n----- finished function runTest3_allGenePrediction_dimReduction_NMF -----\n")

    pass


def runExperiment4_allGenePrediction_dimReduction_AutoEncoder(dataset : Dataset, device):

    print("\n----- entered function runTest4_allGenePrediction_dimReduction_AutoEncoder -----")

    
    

    print("\n----- finished function runTest4_allGenePrediction_dimReduction_AutoEncoder -----\n")

    pass