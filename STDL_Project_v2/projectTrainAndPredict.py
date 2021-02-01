import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torchvision
import numpy as np
import pandas as pd
from projectModels import *
from projectUtilities import *

import matplotlib

matplotlib.use('Agg') # TODO: delete later if you want to use plot in jupyter notebook

from matplotlib import pyplot as plt
import seaborn as sns
from numpy import savetxt


###def train_prediction_model(model_to_train, ds_train, dl_train, loss_fn, optimizer, num_of_epochs_wanted, max_alowed_number_of_batches, device): ### TODO delete if not needed
def train_prediction_model(model_to_train, ds_train, loss_fn, optimizer, hyperparams, model_name, dataset_name, device):
    '''
    This is the main function for training our models.
    '''
    print("/ * \ ENTERED train_prediction_model / * \ ")
    '''
    preparations
    '''
    # for me - name changing
    num_of_epochs = hyperparams['num_of_epochs']
    max_alowed_number_of_batches = hyperparams['max_alowed_number_of_batches']
    model = model_to_train

    # create a SHUFFLING (!) dataloader
    dl_train = DataLoader(ds_train, batch_size=hyperparams['batch_size'], num_workers=hyperparams['num_workers'] , shuffle=True)  # NOTE: shuffle = TRUE !!!

    # important: load model to cuda
    if device.type == 'cuda':
        model = model.to(device=device) 
    
    # compute actual number of batches to train on in each epoch
    num_of_batches = (len(ds_train) // dl_train.batch_size)
    if num_of_batches > max_alowed_number_of_batches:
        print(f'NOTE: in order to speed up training (while damaging accuracy) the number of batches per epoch was reduced from {num_of_batches} to {max_alowed_number_of_batches}')
        num_of_batches = max_alowed_number_of_batches
    else:
        # make sure there are no leftover datapoints not used because of "//"" calculation above
        if (len(ds_train) % dl_train.batch_size) != 0:
            num_of_batches = num_of_batches + 1  

    '''
    BEGIN TRAINING !!!
    # note 2 loops here: external (epochs) and internal (batches)
    '''    
    print("****** begin training ******")

    loss_value_averages_of_all_epochs = []

    for iteration in range(num_of_epochs):
        print(f'iteration {iteration+1} of {num_of_epochs} epochs') # TODO: comment this line if  working on notebook
        
        # init variables for external loop
        dl_iter = iter(dl_train)  # iterator over the dataloader. called only once, outside of the loop, and from then on we use next() on that iterator
        loss_values_list = []

        for batch_index in range(num_of_batches):
            #print(f'iteration {iteration+1} of {num_of_epochs} epochs: batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            #                                                                                                                         # NOTE: this only works in the notebook - # TODO uncomment this line when working on notebook
            # get current batch data 
            data = next(dl_iter)  # note: "data" variable is a list with 3 elements
            # x, y, _ = data  # TODO: NOTE this is changed in 191120 to the following 2 lines below
                            # note :  x.shape is: torch.Size([25, 3, 176, 176]) y.shape is: torch.Size([25]) because the batch size is 25.
                            # NOTE that the third argument recieved here is "column" and is not currently needed
            x = data[0]  # TODO NOTE: changed in 191120 because there is no 3rd argument in the new mandalay version
            y = data[1]  # TODO NOTE: changed in 191120 because there is no 3rd argument in the new mandalay version

            
            # load to device
            if device.type == 'cuda':
                x = x.to(device=device)  
                y = y.to(device=device)
            
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)  
            
            # load to device
            y_pred = y_pred.squeeze() #NOTE !!!!!!! probably needed for the single gene prediction later on

            # Compute (and save) loss.
            loss = loss_fn(y_pred, y) 
            loss_values_list.append(loss.item())

            # Before the backward pass, use the optimizer object to zero all of the gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are accumulated in buffers( i.e, not overwritten) 
            # whenever ".backward()" is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            # delete unneeded tesnors from GPU to save space
            del x, y

        ##end of inner loop
        # print(f'\nfinished inner loop.')

        # data prints on the epoch that ended
        # print(f'in this epoch: min loss {np.min(loss_values_list)} max loss {np.max(loss_values_list)}')
        # print(f'               average loss {np.mean(loss_values_list)}')
        average_value_this_epoch = np.mean(loss_values_list)
        # print(f'in this epoch: average loss {average_value_this_epoch}')
        loss_value_averages_of_all_epochs.append(average_value_this_epoch)

 
    print(f'finished all epochs !                                         ')  # spaces ARE intended
    print(f'which means, that this model is now trained.')
    
    print(f'plotting the loss convergence for the training of this model: ')
    plot_loss_convergence(loss_value_averages_of_all_epochs, model_name, dataset_name) #TODO: temporarily commented during 201120 mandalay data experimentation (this function occurs more times than before and each time runs ove existing object)

    print(" \ * / FINISHED train_prediction_model \ * / ")

    return model


def getSingleDimPrediction(dataset, model, device):
    '''
    REMINDER:
    in 1 dim  prediction experiment we chose a single gene from matrix_df
    and then we trained the model to predict a 1-dimensional vector  meaning - to y values for that gene

              Predict(one_image) = single y value

    THIS FUNCTION:
    we will test our model on all of the images from the test dataset !
    we will predict:
            Predict(all_images) = vector of size (all_images_amount x 1)

    lets denote M_truth to be the test dataframe matrix that contains all K values.
    if we denote that vector as  y_pred == M_pred  then we will want to compare between:

                                 M_pred ~ M_truth

    also, since these should be both 1-d vectors, we will perform a scatter plot of the result !
    '''

    print("\n----- entered function getSingleDimPrediction -----")

    '''
    prepare the data
    '''
    
    '''
    prepare the data
    '''
    # define the batch size for the dataloader
    batch_size = 25
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  #NOTE: important !!! shuffle here MUST be false or all order is lost !!!
    
    # check if dataset is augmented (=has x8 more images than the original dataset). 
    # note that if the dataset is augmented, we are talking about the training dataset - and if not it is the testing dataset
    # if the dataset is augmented, then we need to use here (= in the experiment and NOT in training) only the UNAUGMENTED images.
    # ASSUMPTION: the augmented dataset's image folder object is a concatanation dataset created by me. 
    #             the first `dataset.num_of_images_with_no_augmentation` images from it are the only ones i need for the experiment.
    num_of_batches = 0
    augmented_flag = False
    remainder = -1
    if dataset.size_of_dataset != dataset.num_of_images_with_no_augmentation:  # meaning this dataset is augmented, meaning this is a TRAIN dataset
        num_of_batches = (dataset.num_of_images_with_no_augmentation // batch_size)
        augmented_flag = True
        if (dataset.num_of_images_with_no_augmentation % batch_size) != 0:
            num_of_batches = num_of_batches + 1
            remainder = dataset.num_of_images_with_no_augmentation % batch_size
    else:  # meaning this dataset is NOT augmented, meaning this is a TEST dataset
        num_of_batches = (len(dataset) // batch_size)
        if (len(dataset) % batch_size) != 0:
            num_of_batches = num_of_batches + 1
         
    # create the dataloader
    dl_iter = iter(dataloader)
    # define an empty variable for the model's forward pass iterations
    y_pred_final = None

    for batch_index in range(num_of_batches):
        with torch.no_grad():  # no need to keep track of gradients since we are only using the forward of the model
            #print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            #      \r doesnt work on a text file
            data = next(dl_iter)

            #x, _, _ = data  # x.shape should be (all_images_size, 3, 176, 176)  # TODO: NOTE this is changed in 191120 to the following line below
            x = data[0]                                                          # TODO NOTE: changed in 191120 because there is no 3rd argument in the new mandalay version

            # small correction if this is an augmented dataset and this is the LAST batch ... we dont want the augmented images
            if batch_index == num_of_batches-1 and augmented_flag is True and remainder != -1:
                split_result = torch.split(tensor=x, split_size_or_sections=remainder, dim=0)  # dim=0 means split on batch size
                x = split_result[0]

            # load to device
            if device.type == 'cuda':
                x = x.to(device=device)  

            '''
            feed data to model to get K dim result
            '''
            # # This nex lines is to heavy for the GPU (apparantly); and so it is divided into small portions
            y_pred = model(x)

            if y_pred_final is None:  # means this is the first time the prediction occured == first iteration of the loop
                y_pred_final = y_pred.cpu().detach().numpy()
            else:               # means this is not the first time 
                                # in that case, we will "stack" (concatanate) numpy arrays
                                # np.vstack: # Stack arrays in sequence vertically (row wise) !
                y_pred_curr_prepared = y_pred.cpu().detach().numpy()
                y_pred_final = np.vstack((y_pred_final, y_pred_curr_prepared))
                       
            # delete vectors used from the GPU
            del x
            # finished loop

    '''
    ***
    '''
    # get M_pred
    M_pred = y_pred_final 
    # get M_truth
    M_truth = dataset.reduced_dataframe.to_numpy()  # this is the full 33538 dimensional vector (or 18070~ after reduction) from the original dataframe
    # assert equal sizes
    M_truth = M_truth.transpose()  #NOTE the transpose here to match the shapes !!!
    M_pred = M_pred.squeeze()
    M_truth = M_truth.squeeze()
    print(f'M_pred.shape {M_pred.shape} ?==? M_truth.shape {M_truth.shape}')
    assert M_pred.shape == M_truth.shape

    ### final print
    print(f'Reached end of the function, printing information about the prediction vs the truth values')
    temp_df = pd.DataFrame({'M_truth':M_truth, 'M_pred':M_pred})
    print(temp_df)

    print("\n----- finished function getSingleDimPrediction -----")
    #
    return M_truth, M_pred





