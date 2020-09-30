import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torchvision
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from deepNetworkArchitechture import ConvNet, AutoencoderNet
from projectUtilities import compare_matrices, calculate_distance_between_matrices, printInfoAboutDataset, plot_Single_Gene_PredAndTrue_on_LargeImage
from matplotlib import pyplot as plt


def train_prediction_model(model_to_train, ds_train, dl_train, loss_fn, optimizer, num_of_epochs_wanted, max_alowed_number_of_batches, device):
    '''
    This is the main function for training our models.
    '''
    print("/ * \ ENTERED train_prediction_model / * \ ")
    '''
    preparations
    '''
    # for me
    num_of_epochs = num_of_epochs_wanted
    model = model_to_train
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
            num_of_batches = num_of_batches + 1  #TODO: verify

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

        for batch_index in range(num_of_batches):
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            # get current batch data 
            data = next(dl_iter)  # note: "data" variable is a list with 2 elements:  data[0] is: <class 'torch.Tensor'> data[1] is: <class 'torch.Tensor'>
            x, y, _ = data  # note :  x.shape is: torch.Size([25, 3, 176, 176]) y.shape is: torch.Size([25]) because the batch size is 25. 
                            # NOTE that the third argument recieved here is "column" and is not currently needed
            
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

        #end of inner loop
        print(f'\nfinished inner loop.')


        # data prints on the epoch that ended
        # print(f'in this epoch: min loss {np.min(loss_values_list)} max loss {np.max(loss_values_list)}')
        # print(f'               average loss {np.mean(loss_values_list)}')
        print(f'in this epoch: average loss {np.mean(loss_values_list)}')

 
    print(f'finished all epochs !')
    print(f'which means, that this model is now trained.')
    print(" \ * / FINISHED train_prediction_model \ * / ")
    return model


def runExperiment(ds_train : Dataset, ds_test : Dataset, hyperparams, device, model_name, dataset_name):
    '''
    **runExperimentWithModel_BasicConvNet**
    this function performs 2 things:
    (1) Trains the model on patient 1 data (train data)
    (2) Tests the model  on patient 2 data (test data)
    '''
    print("\n----- entered function runExperiment -----")

    '''
    prepare model, loss and optimizer instances
    '''
    # create the model
    model = get_model_by_name(name=model_name, dataset=ds_train, hyperparams=hyperparams)
    # create the loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    '''
    Train the model
    '''
    dl_train = DataLoader(ds_train, hyperparams['batch_size'], shuffle=True)
    trained_model = train_prediction_model(model_to_train=model, ds_train=ds_train, dl_train=dl_train, loss_fn=loss_fn, 
                                            optimizer=optimizer, num_of_epochs_wanted=hyperparams['num_of_epochs'], 
                                            max_alowed_number_of_batches=hyperparams['max_alowed_number_of_batches'],
                                            device=device)
    
    '''
    Test the model and print comparisons
    '''
    ### important NOTE!  first 2 exps use ONLY ds_test, the 2 last ones use both train and test !!!!!!!!!!!! 

    if dataset_name.startswith("single_gene"):
        M_truth, M_pred = getSingleDimPrediction(dataset=ds_test, model=trained_model, device=device, model_name=model_name, dataset_name=dataset_name) # note that this function saves a figure
        baseline = np.full(shape=M_truth.shape, fill_value=np.average(M_truth))  # `full` creates an array of wanted size where all values are the same fill value
        compare_matrices(M_truth, M_pred, Baseline=baseline)
        
    elif dataset_name.startswith("k_genes"):
        M_truth, M_pred = getKDimPrediction(dataset=ds_test, model=trained_model, device=device)
        baseline = np.full(shape=M_truth.shape, fill_value=np.average(M_truth))  # `full` creates an array of wanted size where all values are the same fill value
                                                                                 # note - since no axis is given, `np.average` performs an average over all elements in all dimensions
        compare_matrices(M_truth, M_pred, Baseline=baseline)

    elif dataset_name.startswith("NMF"):
        ## perform comparisons on train data if this is not the augmented DS
        M_truth, M_pred = getFullDimsPrediction_with_NMF_DS(dataset=ds_train, W=ds_train.W, model=trained_model, device=device)
        # train-error comparisons: M_truth ~ M_fast_reconstruction ~ M_pred
        #                      orig_matrix ~       W * H           ~ W * H_pred
        M_fast_reconstruction = np.matmul(ds_train.W, ds_train.H)
        compare_matrices(M_truth, M_pred, Baseline=M_fast_reconstruction)
        
        # perform on test data
        M_truth, M_pred = getFullDimsPrediction_with_NMF_DS(dataset=ds_test, W=ds_train.W, model=trained_model, device=device)
        # test-error comparisons: M_truth ~ M_pred
        #                     orig_matrix ~ W * H_pred
        compare_matrices(M_truth, M_pred, Baseline=None)
        
    elif dataset_name.startswith("AE"):
        ## perform comparisons on train data if this is not the augmented DS
        M_truth, M_pred = getFullDimsPrediction_with_AE_DS(dataset=ds_train, AEnet=ds_train.autoEncoder, model=trained_model, device=device)
        # train-error comparisons: M_truth ~ M_fast_reconstruction ~ M_pred
        #                      orig_matrix ~  Decode(Encode(M))    ~ Decode(Predict(X))
        M_fast_reconstruction = getAutoEncoder_M_fast_reconstruction(dataset=ds_train, model=trained_model, device=device)
        compare_matrices(M_truth, M_pred, Baseline=M_fast_reconstruction)

        # perform on test data
        M_truth, M_pred = getFullDimsPrediction_with_AE_DS(dataset=ds_test, AEnet=ds_train.autoEncoder, model=trained_model, device=device)
        # test-error comparisons: M_truth ~ M_pred
        #                     orig_matrix ~ W * H_pred
        compare_matrices(M_truth, M_pred, Baseline=None)
        
    # delete unneeded tesnors from GPU to save space
    del trained_model
    # goodbye
    print("\n----- finished function runExperiment -----")
    pass


def getSingleDimPrediction(dataset, model, device, model_name, dataset_name):
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
    batch_size = 20
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  #NOTE: important !!! shuffle here MUST be false or all order is lost !!!
    dl_iter = iter(dataloader)
    num_of_batches = (len(dataset) // batch_size)
    if (len(dataset) % batch_size) != 0:
        num_of_batches = num_of_batches + 1

    y_pred_final = None

    for batch_index in range(num_of_batches):
        with torch.no_grad():  # no need to keep track of gradients since we are only using the forward of the model
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            data = next(dl_iter)
            x, _, _ = data  # x.shape should be (all_images_size, 3, 176, 176)

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
    assert M_pred.shape == M_truth.shape

    '''
    plot results and save them
    '''
    plt.clf()  # clears previous plots
    # create a scatter
    plt.scatter(x=M_truth, y=M_pred, label='M_truth VS M_pred')
    # create a line
    x = np.linspace(-0.5,3.5,100) # linspace() function to create evenly-spaced points in a given interval
    y = x
    plt.plot(x, y, '--k', label='y=x plot') # create a line # "--k" means black dashed line
    # set surroundings
    plt.xlabel(f'M_truth values')
    plt.ylabel(f'M_pred values')
    plt.title(f'Result of comparison between M_truth VS M_pred\nwith model: {model_name} on dataset {dataset_name}')
    plt.legend()
    filename = f'{dataset_name}_{model_name}_comparison'
    plt.savefig(f'{filename}.png', bbox_inches='tight')

    # 290920 testing TODO: maybe delete later
    temp_df = pd.DataFrame({'M_truth':M_truth, 'M_pred':M_pred})
    print(f'info about M_truth VS M_pred dataframe: \n{temp_df}\n{temp_df.info()}')
    print(f'some more info:::: truth first, then pred')
    # from projectUtilities import printInfoAboutReducedDF
    # printInfoAboutReducedDF(temp_df['M_truth'].to_frame())
    # printInfoAboutReducedDF(temp_df['M_pred'].to_frame())
    temp_df.plot('M_truth', 'M_pred', kind='scatter')
    temp_df.to_csv(f'{filename}.csv')

    # perform plot on larger image
    plot_Single_Gene_PredAndTrue_on_LargeImage(dataset, M_pred, M_truth) 

    print("\n----- finished function getSingleDimPrediction -----")
    #
    return M_truth, M_pred


def getKDimPrediction(dataset, model, device):
    '''
    REMINDER:
    in K dim  prediction experiment we chose the K genes with the highest variance from matrix_df
    and then we trained the model to predict a k-dimensional vector  meaning - to predict k genes (for a single image)

              Predict(one_image) = vector of size (1 x K)

    THIS FUNCTION:
    we will test our model on the K-genes with the highest varience from the test dataset !
    we will predict:
            Predict(all_images) = vector of size (all_images_amount x K)

    lets denote M_truth to be the test dataframe matrix that contains all K values.
    if we denote that vector as  y_pred == M_pred  then we will want to compare between:

                                 M_pred ~ M_truth
                            
    '''
    print("\n----- entered function getKDimPrediction -----")

    '''
    prepare the data
    '''

    batch_size = 20
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  #NOTE: important !!! shuffle here MUST be false or all order is lost !!!
    dl_iter = iter(dataloader)
    num_of_batches = (len(dataset) // batch_size)
    if (len(dataset) % batch_size) != 0:
        num_of_batches = num_of_batches + 1

    y_pred_final = None

    for batch_index in range(num_of_batches):
        with torch.no_grad():  # no need to keep track of gradients since we are only using the forward of the model
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            data = next(dl_iter)
            x, _, _ = data  # x.shape should be (all_images_size, 3, 176, 176)

            # load to device
            if device.type == 'cuda':
                x = x.to(device=device)  
        
            '''
            feed data to model to get K dim result
            '''
            # # This next line is to heavy for the GPU (apparantly); and so it is divided into small portions
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
    M_truth = dataset.reduced_dataframe.to_numpy()  # this is the full 33538 dimensional vector (or 23073~ after reduction) from the original dataframe
    # assert equal sizes
    M_truth = M_truth.transpose()  #NOTE the transpose here to match the shapes !!!
    M_pred = M_pred.squeeze()
    M_truth = M_truth.squeeze()
    assert M_pred.shape == M_truth.shape


    print("\n----- finished function getKDimPrediction -----")
    #
    return M_truth, M_pred


def getFullDimsPrediction_with_NMF_DS(dataset, W, model, device):
    '''
    REMINDER:
    NMF decomposition performs  M = W * H
    lets denote M == M_truth

    THIS FUNCTION:
    this function will perform dimension restoration using matrix multiplication:
    if we denote  y_pred == H_pred  then:
                            W * H_pred = M_pred

    and then if we want we can compare   M_pred  to   M_truth

    NOTE !!! in both cases of `dataset` being training or testing dataset,
                W should be from the TRAINING dataset and NOT FROM the TESTING dataset !!!
    '''
    print("\n----- entered function getFullDimsPrediction_with_NMF_DS -----")

    '''
    prepare the data
    '''
    
    # define the batch size for the dataloader
    batch_size = 20
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
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            data = next(dl_iter)
            x, _, _ = data  # x.shape should be (all_images_size, 3, 176, 176)

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
            # # This next line is to heavy for the GPU (apparantly); and so it is divided into small portions
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
    restore dimension to y_pred by performing: res =      W * y_pred                =      M_pred
                                                (33538 x K) * (K * num_of_images)   = (33538 x num_of_images) 
                                                the number 33538 might change due to pre-processing steps
    '''
    # both vecotors need a little preparation for the multiplication
    y_pred_prepared = y_pred_final.transpose() #note the transpose here !
    W_prepared = W

    # get M_pred
    # # NOTE: the cuda tesnor needs a little conversion first from cuda to cpu and to the right dimension
    M_pred = np.matmul(W_prepared, y_pred_prepared)  

    # get M_truth
    M_truth = dataset.matrix_dataframe.to_numpy()  # this is the full 33538 dimensional vector (or 23073~ after reduction) from the original dataframe
    # assert equal sizes
    M_pred = M_pred.squeeze()
    M_truth = M_truth.squeeze()
    assert M_pred.shape == M_truth.shape


    print("\n----- finished function getKDimPrediction -----")

    #
    return M_truth, M_pred


def getFullDimsPrediction_with_AE_DS(dataset, AEnet, model, device):
    '''
    REMINDER:
    we recieved our y values from the latent space using the pre-trained encoder
    y_pred was recieved due to: 
    note that  y_pred == Predict(X)

    THIS FUNCTION:
    this function will perform dimension restoration using the decoder
    we denote  Decode(y_pred) == Decode(Predict(X)) == M_pred
    and M_truth as the original matrix
                            
    and then if we want we can compare   M_pred  to   M_truth

    NOTE !!! in both cases of `dataset` being training or testing dataset,
             AEnet should be from the TRAINING dataset and NOT FROM the TESTING dataset !!!
    '''
    print("\n----- entered function getFullDimsPrediction_with_AE_DS -----")

    '''
    prepare the data
    '''
    # printInfoAboutDataset(dataset)

    batch_size = 1  ### NOTE: Important !!! the reason this is 1 it to correlate with the way we trained the AE net... see "return_trained_AE_net" in loadAndPreProcess.py .
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

    '''
    run the model on our data,
    and when the model is done, decode the result.
    the decoded results will be stacked together - and will add up to become M_pred
    '''

    for batch_index in range(num_of_batches):
        with torch.no_grad():  # no need to keep track of gradients since we are only using the forward of the model
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            data = next(dl_iter)
            x, _, _ = data  # x.shape should be (all_images_size, 3, 176, 176)

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
            y_pred = model(x)

            '''
            get the decoded version of the network's output, conver it to numpy on the cpu, and stack it for later usage
            '''
            y_pred_decoded = AEnet.decodeWrapper(y_pred)

            if y_pred_final is None:  # means this is the first time the prediction occured == first iteration of the loop

                y_pred_final = y_pred_decoded.cpu().detach().numpy()
            else:               # means this is not the first time 
                                # in that case, we will "stack" (concatanate) numpy arrays
                                # np.vstack: # Stack arrays in sequence vertically (row wise) !
                y_pred_curr_prepared = y_pred_decoded.cpu().detach().numpy()
                y_pred_final = np.vstack((y_pred_final, y_pred_curr_prepared))
            
            # delete vectors used from the GPU
            del x
            # finished loop
    
    '''
    restore dimension to y_pred by using the decoder
    '''
    # restore dimension to y_pred by using the decoder (from our pre-trained autoencoder)
    M_pred = y_pred_final.transpose() #NOTE the transpose - it is used to allign shapes with M_truth

    # get M_truth
    M_truth = dataset.matrix_dataframe.to_numpy()  # this is the full 33538 dimensional vector (or 23073~ after reduction) from the original dataframe
    
    # assert equal sizes
    M_pred = M_pred.squeeze()
    M_truth = M_truth.squeeze()
    assert M_pred.shape == M_truth.shape

    print("\n----- finished function getFullDimsPrediction_with_AE_DS -----")
    #
    return M_truth, M_pred
    

def getAutoEncoder_M_fast_reconstruction(dataset, model, device):
    '''
    perform a fast decoding and encoding to our matrix dataframe using the trained AE net - without any other training
    '''
    print("\n----- entered function getAutoEncoder_M_fast_reconstruction -----")

    '''
    preparations
    '''
    batch_size = 1  ### NOTE: Important !!! the reason this is 1 it to correlate with the way we trained the AE net... see "return_trained_AE_net" in loadAndPreProcess.py .
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
    result = None

    '''
    run the model on our data,
    and when the model is done, decode the result.
    the decoded results will be stacked together - and will add up to become M_pred
    '''

    for batch_index in range(num_of_batches):
        with torch.no_grad():  # no need to keep track of gradients since we are only using the forward of the model
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            data = next(dl_iter)
            x = data  # x.shape should be (all_images_size, 3, 176, 176)
            x = x.float()  # needed to avoid errors of conversion

            # small correction if this is an augmented dataset and this is the LAST batch ... we dont want the augmented images
            if batch_index == num_of_batches-1 and augmented_flag is True and remainder != -1:
                split_result = torch.split(tensor=x, split_size_or_sections=remainder, dim=0)  # dim=0 means split on batch size
                x = split_result[0]

            # load to device
            if device.type == 'cuda':
                x = x.to(device=device)  

            # get the encoded version
            y_pred_encoded = dataset.autoEncoder.encodeWrapper(x)
            # get the decoded version
            y_pred_decoded = dataset.autoEncoder.decodeWrapper(y_pred_encoded)

            if result is None:  # means this is the first time the prediction occured == first iteration of the loop

                result = y_pred_decoded.cpu().detach().numpy()
            else:               # means this is not the first time 
                                # in that case, we will "stack" (concatanate) numpy arrays
                                # np.vstack: # Stack arrays in sequence vertically (row wise) !
                y_pred_curr_prepared = y_pred_decoded.cpu().detach().numpy()
                result = np.vstack((result, y_pred_curr_prepared))
            
            # delete vectors used from the GPU
            del x
            # finished loop
    
    '''
    finish up
    '''
    M_fast_reconstruction = result.transpose() #NOTE the transpose - it is used to allign shapes with M_truth

    # assert equal sizes
    assert M_fast_reconstruction.shape == dataset.matrix_dataframe.to_numpy().shape

    print("\n----- finished function getAutoEncoder_M_fast_reconstruction -----")
    #
    return M_fast_reconstruction


def get_model_by_name(name, dataset, hyperparams):
    '''
    prep:
    '''
    x0, y0, _ = dataset[0]  # NOTE that the third argument recieved here is "column" and is not currently needed
    in_size = x0.shape # note: if we need for some reason to add batch dimension to the image (from [3,176,176] to [1,3,176,176]) use x0 = x0.unsqueeze(0)  # ".to(device)"
    output_size = 1 if isinstance(y0, int) or isinstance(y0, float) else y0.shape[0] # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)
    '''
    get_model_by_name
    '''
    if name == 'BasicConvNet':
        model = ConvNet(in_size, output_size, channels=hyperparams['channels'], pool_every=hyperparams['pool_every'], hidden_dims=hyperparams['hidden_dims'])
        return model
    elif name == 'DensetNet121':
        # create the model from an existing architechture
        # explanation of all models in: https://pytorch.org/docs/stable/torchvision/models.html
        model = torchvision.models.densenet121(pretrained=False)

        # update the exisiting model's last layer
        input_size = model.classifier.in_features
        output_size = 1 if isinstance(y0, int) or isinstance(y0, float) else y0.shape[0] # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)  !!! 
                
        model.classifier = torch.nn.Linear(input_size, output_size, bias=True)
        model.classifier.weight.data.zero_()
        model.classifier.bias.data.zero_()
        return model
    elif name == 'Inception_V3':
        # create the models
        # explanation of all models in: https://pytorch.org/docs/stable/torchvision/models.html
        print(f'starting to load the model inception_v3 from torchvision.models. this is quite heavy, and might take some time ...')
        model = torchvision.models.inception_v3(pretrained=False) 
        print(f'finished loading model')

        # update the existing models last layer
        input_size = model.fc.in_features
        output_size = 1 if isinstance(y0, int) or isinstance(y0, float) else y0.shape[0] # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)  !!! 
                                                                
        model.fc = torch.nn.Linear(input_size, output_size, bias=True)
        model.fc.weight.data.zero_()
        model.fc.bias.data.zero_()
        return model


def get_Trained_AEnet(dataset_from_matrix_df, z_dim, num_of_epochs, device):
    '''
    trains the AE net on the matrix dataframe
    returns the trained autoencoder model
    '''

    print("\n----- entered function return_trained_AE_net -----")

    '''
    prep our dataset and dataloaders
    '''
    batch_size = 1  # this number was reduced because the server was busy and i got CUDA OUT OF MEMORY. need to increase later
                    # IMPORTANT NOTE ON THE BATCH SIZE !!!
                    # at first it was a high number, but due to lack of memory was reduced to 5, which worked.
                    # then, I found out that if the batch size is not 1, i cannot use the networks encoder network inside AEnet
                    # because is expects 33K features * 5 batch_size as input - but in __get_item__ (which is the entire end goal of our current method)
                    # we only get one item at a time.... and so - batch size was changed to 1.
                    # later on, this can be improved.
    dataset = dataset_from_matrix_df
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    x0 = dataset[0]
    num_of_features = len(x0)

    '''
    prepare model, loss and optimizer instances
    '''

    # model
    connected_layers_dim_list = [100*z_dim, 10*z_dim, 5*z_dim]  #NOTE: this is without the first and last layers ! # TODO: change code to get this value from user ? (from the notebook with the hyperparams dictionary)
    print(f'note - number of (hidden) linear layers is supposed to be {len(connected_layers_dim_list)}')
    model = AutoencoderNet(in_features=num_of_features, connected_layers_dim_list=connected_layers_dim_list, z_dim=z_dim, batch_size=batch_size, device=device)

    if device.type == 'cuda':
        model = model.to(device=device)  
    
    # loss and optimizer
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-4      # TODO: change code to get this value from user ? (from the notebook with the hyperparams dictionary)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    '''
    now we can perform the training
    '''

    print("****** begin training ******")
    num_of_epochs = 5       # TODO: change code to get this value from user ? (from the notebook with the hyperparams dictionary)
    max_alowed_number_of_batches = 999999  # the purpose of this var is if i dont realy want all of the batches to be trained uppon ... 
    num_of_batches = (len(dataset) // batch_size)  
    if num_of_batches > max_alowed_number_of_batches:
        print(f'NOTE: in order to speed up training (while damaging accuracy) the number of batches per epoch was reduced from {num_of_batches} to {max_alowed_number_of_batches}')
        num_of_batches = max_alowed_number_of_batches
    else:
        # make sure there are no leftover datapoints not used because of "//"" calculation above
        if (len(dataset) % batch_size) != 0:
            num_of_batches = num_of_batches + 1  #TODO: verify 


    # note 2 loops here: external and internal
    for iteration in range(num_of_epochs):
        print(f'\niteration {iteration+1} of {num_of_epochs} epochs')
        
        # init variables for external loop
        dl_iter = iter(dataloader)  # iterator over the dataloader. called only once, outside of the loop, and from then on we use next() on that iterator
        loss_values_list = []

        for batch_index in range(num_of_batches):
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            # get current batch data 
            data = next(dl_iter)  # note: "data" variable is a list with 2 elements:  data[0] is: <class 'torch.Tensor'> data[1] is: <class 'torch.Tensor'>
            #
            x = data  # note :  x.shape is: torch.Size([25, 3, 176, 176]) y.shape is: torch.Size([25]) because the batch size is 25
            x = x.float()  # needed to avoid errors of conversion
            if device.type == 'cuda':
                x = x.to(device=device)  

            # Forward pass: compute predicted y by passing x to the model.
            x_reconstructed = model(x)  
            if device.type == 'cuda':
                x_reconstructed = x_reconstructed.to(device=device)
            
        
            # Compute (and print) loss.
            loss = loss_fn(x_reconstructed, x)  
            loss_values_list.append(loss.item())

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
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

    pass
    print("\n----- finished function return_trained_AE_net -----\n")

    # return the trained model
    return model


