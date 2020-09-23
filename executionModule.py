import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torchvision
import numpy as np
from sklearn.decomposition import NMF
from deepNetworkArchitechture import ConvNet
from projectUtilities import calculate_distance_between_matrices


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
    
    # print info for user:
    # print(f'recieved model {model}\nrecieved loss_fn {loss_fn}\nrecieved optimizer {optimizer}\nrecieved num_of_epochs_wanted {num_of_epochs_wanted}\nrecieved max_alowed_number_of_batches {max_alowed_number_of_batches}')

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

        for batch_index in range(num_of_batches):
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            # get current batch data 
            data = next(dl_iter)  # note: "data" variable is a list with 2 elements:  data[0] is: <class 'torch.Tensor'> data[1] is: <class 'torch.Tensor'>
            x, y, _ = data  # note :  x.shape is: torch.Size([25, 3, 176, 176]) y.shape is: torch.Size([25]) because the batch size is 25. 
                            # NOTE that the third argument recieved here is "column" and is not currently needed
            
            # load to device
            if device.type == 'cuda':
                x = x.to(device=device)  
                y = y.float()
                y = y.to(device=device)
            
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)  
            
            # load to device
            if device.type == 'cuda':
                y_pred = y_pred.float()
                y_pred = y_pred.squeeze()
                y_pred = y_pred.to(device=device)

            # checking for same values in the predcition y_pred as in ground truth y:
            # first, flatten the vectors, and ROUND THE VALUES ! NOTE !!!!!
            y_pred_rounded_flattened = torch.round(y_pred).type(torch.int).flatten()  #TODO: note this rounding process to the closest iteger !
            y_rounded_flattened = torch.round(y).type(torch.int).flatten()  #TODO: note this rounding process to the closest iteger !
            # print(f'--delete-- y.shape {y.shape}, y_pred.shape {y_pred.shape}, y_pred_rounded.shape {y_pred_rounded.shape}')

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

            # delete unneeded tesnors from GPU to save space #TODO: need to check that this code works
            del x, y, y_pred

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
    print("\n----- entered function runExperimentWithModel_BasicConvNet -----")

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
    ### important NOTE!  first 2 exps use ONLY ds_test, the 2 last ones use both train and test !!!!!!!!!!!! TODO verify this is OK

    if dataset_name == 'single_gene':
        M_truth, M_pred = getSingleDimPrediction(dataset=ds_test, model=trained_model, device=device, model_name=model_name) # note that this function saves a figure
        compare_matrices(M_truth, M_pred, None)
        
    elif dataset_name == 'k_genes':
        M_truth, M_pred = getKDimPrediction(dataset=ds_test, model=trained_model, device=device)
        compare_matrices(M_truth, M_pred, None)

    elif dataset_name == 'NMF':
        M_truth, M_pred = getFullDimsPrediction_with_NMF_DS(dataset=ds_train, model=trained_model, device=device)
        # train-error comparisons: M_truth ~ M_fast_reconstruction ~ M_pred
        #                      orig_matrix ~       W * H           ~ W * H_pred
        M_fast_reconstruction = np.mm(ds_train.W, ds_train.H)
        compare_matrices(M_truth, M_pred, M_fast_reconstruction)

        M_truth, M_pred = getFullDimsPrediction_with_NMF_DS(dataset=ds_test, model=trained_model, device=device)
        # test-error comparisons: M_truth ~ M_pred
        #                     orig_matrix ~ W * H_pred
        compare_matrices(M_truth, M_pred)
        
    elif dataset_name == 'AE':
        result_train_data = getFullDimsPrediction_with_AE_DS(dataset=ds_train, model=trained_model, device=device)
        # train-error comparisons: M_truth ~ M_fast_reconstruction ~ M_pred
        #                      orig_matrix ~  Decode(Encode(M))    ~ Decode(Predict(X))
        M_fast_reconstruction = np.mm(ds_train.W, ds_train.H)
        compare_matrices(M_truth, M_pred, M_fast_reconstruction)

        result_test_data = getFullDimsPrediction_with_AE_DS(dataset=ds_test, model=trained_model, device=device)
        # test-error comparisons: M_truth ~ M_pred
        #                     orig_matrix ~ W * H_pred
        compare_matrices(M_truth, M_pred)
        
    # delete unneeded tesnors from GPU to save space
    del trained_model
    # goodbye
    print("\n----- finished function runExperimentWithModel_BasicConvNet -----")
    pass


def compare_matrices(M_truth, M_pred, M_fast_reconstruction=None): #note the None if not needed
    # TODO: might need to move to utilities
    print(f'TODO: print comparison of error results')  #TODO !
    '''
    method 1 - compare identical elements
    '''
    # # trick to get num of identical elements between 2 NUMPY arrays 
    # number_of_identical_elements = np.sum(y0_groundtruth_prepared == y0_pred_prepared) 
    # print(f'corret prediction after dimensionality restoration: {number_of_identical_elements} out of {len(y0_ground_truth_all_dims)}')
    '''
    method 2 - calculate distance between matrices
    '''
    error1 = calculate_distance_between_matrices(M_truth, M_pred)
    error2 = calculate_distance_between_matrices(M_truth, M_fast_reconstruction)
    error3 = calculate_distance_between_matrices(M_pred, M_fast_reconstruction)
    if M_fast_reconstruction is None:
        print(f'recieved M_fast_reconstruction=None. errors with it will be 0')
    print(f'distance between M_truth, M_pred: {error1}')
    print(f'distance between M_truth, M_fast_reconstruction: {error2}')
    print(f'distance between M_pred, M_fast_reconstruction: {error3}')

    pass


def getSingleDimPrediction(dataset, model, device, model_name):
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    dl_iter = iter(dataloader)
    num_of_batches = (len(dataset) // batch_size)
    if (len(dataset) % batch_size) != 0:
        num_of_batches = num_of_batches + 1

    y_pred_final = None

    for batch_index in range(num_of_batches):
        with torch.no_grad():  # no need to keep track of gradients since we are only using the forward of the model
            print(f'batch {batch_index+1} of {num_of_batches} batches', end='\r') # "end='\r'" will cause the line to be overwritten the next print that comes
            data = next(dl_iter)
            x, y, _ = data  # x.shape should be (all_images_size, 3, 176, 176)
            # print(f'\n--delete-- x.shape {x.shape}')

            # load to device
            if device.type == 'cuda':
                x = x.to(device=device)  
                y = y.float()
                y = y.to(device=device)
        
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
            del y 
            # finished loop


    '''
    ***
    '''
    # both vecotors need a little preparation for the multiplication
    #y_pred_prepared = y0_pred.t().to(device=device) #note the transpose here ! # TODO: this is the version before 230920

    # get M_pred
    # # NOTE: the cuda tesnor needs a little conversion first from cuda to cpu and to the right dimension
    # M_pred = y_pred.cpu().detach().numpy() # TODO: this is the version BEFORE 230920
    M_pred = y_pred_final # TODO: this is the version AFTER 230920
    # get M_truth
    M_truth = dataset.reduced_dataframe.to_numpy()  # this is the full 33538 dimensional vector (or 23073~ after reduction) from the original dataframe
    # assert equal sizes
    M_truth = M_truth.transpose()  #NOTE the transpose here to match the shapes !!!
    print(f'--delete-- verify:  M_pred.shape {M_pred.shape}  ~  M_truth.shape {M_truth.shape}')
    M_pred = M_pred.squeeze()
    M_truth = M_truth.squeeze()
    assert M_pred.shape == M_truth.shape

    # plot results and save them
    from matplotlib import pyplot as plt
    plt.clf()  # clears previous plots
    plt.plot(M_truth, M_pred, label='M_truth VS M_pred')
    plt.xlabel(f'')
    plt.ylabel(f'')
    plt.title(f'Result of comparison between M_truth VS M_pred with model: {model_name}')
    plt.legend()
    filename = f'{model_name}_comparison'
    plt.savefig(f'{filename}.png', bbox_inches='tight')

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

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)  # batch_size=len(dataset) == should be the number of images in the dataset
    dl_iter = iter(dataloader)
    #
    y_pred_final = None

    with torch.no_grad():
        data = next(dl_iter)
        x, y, _ = data  # x.shape should be (all_images_size, 3, 176, 176)
        print(f'--delete-- x.shape {x.shape}')

        # load to device
        if device.type == 'cuda':
            x = x.to(device=device)  
            y = y.float()
            y = y.to(device=device)
        
        '''
        feed data to model to get K dim result
        '''
        # # This nex lines is to heavy for the GPU (apparantly); and so it is divided into small portions
        y_pred_final = model(x)

        # delete vectors used from the GPU
        del x
        del y 
        # finished section of torch.no_grad()
    
    # '''
    # prepare the data
    # '''

    # dataloader = DataLoader(dataset, batch_size=20, shuffle=True)  # batch_size=len(dataset) == should be the number of images in the dataset
    # dl_iter = iter(dataloader)
    # num_of_batches = (len(ds_train) // dl_train.batch_size)

    # y_pred_final = None

    # for batch in range(num_of_batches):
    #     with torch.no_grad():
    #         data = next(dl_iter)
    #         x, y, _ = data  # x.shape should be (all_images_size, 3, 176, 176)
    #         print(f'--delete-- x.shape {x.shape}')

    #         # load to device
    #         if device.type == 'cuda':
    #             x = x.to(device=device)  
    #             y = y.float()
    #             y = y.to(device=device)
        
    #         '''
    #         feed data to model to get K dim result
    #         '''
    #         # # This nex lines is to heavy for the GPU (apparantly); and so it is divided into small portions
    #         y_pred = model(x)

    #         if y_pred_final is None:  # means this is the first time the prediction occured == first iteration of the loop
    #             y_pred_final = y_pred.cpu().detach().numpy()
    #         else:               # means this is not the first time 
    #                             # in that case, we will "stack" (concatanate) numpy arrays
    #                             # np.vstack: # Stack arrays in sequence vertically (row wise) !
    #             y_pred_curr_prepared = y_pred.cpu().detach().numpy()
    #             y_pred_final = np.vstack((y_pred_final, y_pred_curr_prepared))
            
    #         # delete vectors used from the GPU
    #         del x
    #         del y 
    #         # finished loop


    '''
    ***
    '''
    # both vecotors need a little preparation for the multiplication
    #y_pred_prepared = y0_pred.t().to(device=device) #note the transpose here ! # TODO: this is the version before 230920

    # get M_pred
    # # NOTE: the cuda tesnor needs a little conversion first from cuda to cpu and to the right dimension
    # M_pred = y_pred.cpu().detach().numpy() # TODO: this is the version BEFORE 230920
    M_pred = y_pred_final # TODO: this is the version AFTER 230920
    # get M_truth
    M_truth = dataset.matrix_dataframe.to_numpy()  # this is the full 33538 dimensional vector (or 23073~ after reduction) from the original dataframe
    # assert equal sizes
    assert M_pred.shape == M_truth.shape

    #
    return M_truth, M_pred


def getFullDimsPrediction_with_NMF_DS(dataset, model, device):
    '''
    REMINDER:
    NMF decomposition performs  M = W * H
    lets denote M == M_truth

    THIS FUNCTION:
    this function will perform dimension restoration using matrix multiplication:
    if we denote  y_pred == H_pred  then:
                            W * H_pred = M_pred

    and then if we want we can compare   M_pred  to   M_truth
    '''
    print("\n----- entered function getFullDimsPrediction_with_NMF_DS -----")

    '''
    prepare the data
    '''
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)  # batch_size=len(dataset) == should be the number of images in the dataset
    dl_iter = iter(dataloader)
    data = next(dl_iter)
    x, y, _ = data  # x.shape should be (all_images_size, 3, 176, 176)
    print(f'--delete-- x.shape {x.shape}')

    # load to device
    if device.type == 'cuda':
        x = x.to(device=device)  
        y = y.float()
        y = y.to(device=device)
    
    '''
    feed data to model to get K dim result
    '''
    y_pred = model(x)
    print(f'--delete-- y_pred.shape {y_pred.shape}')
    
    '''
    restore dimension to y_pred by performing: res =      W * y_pred                =      M_pred
                                                (33538 x K) * (K * num_of_images)   = (33538 x num_of_images) 
                                                the number 33538 might change due to pre-processing steps
    '''
    # both vecotors need a little preparation for the multiplication
    y_pred_prepared = y0_pred.t().to(device=device) #note the transpose here !
    W_prepared = torch.from_numpy(dataset.W).float().to(device=device)

    print(f'--delete-- verify: W_prepared.shape {W_prepared.shape}, y_pred_prepared.shape {y_pred_prepared.shape}')
    # get M_pred
    # # NOTE: the cuda tesnor needs a little conversion first from cuda to cpu and to the right dimension
    M_pred = torch.mm(W_prepared, y_pred_prepared)  
    M_pred = M_pred.cpu().detach().numpy()
    # get M_truth
    M_truth = dataset.matrix_dataframe.to_numpy()  # this is the full 33538 dimensional vector (or 23073~ after reduction) from the original dataframe
    # assert equal sizes
    assert M_pred.shape == M_truth.shape
    # delete vectors used from the GPU
    del x
    del y 
    #
    return M_truth, M_pred


def getFullDimsPrediction_with_AE_DS(dataset, model, device):
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

    and if wanted we can also compare
    '''
    print("\n----- entered function getFullDimsPrediction_with_NMF_DS -----")

    '''
    prepare the data
    '''
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)  # batch_size=len(dataset) == should be the number of images in the dataset
    dl_iter = iter(dataloader)
    data = next(dl_iter)
    x, y, _ = data  # x.shape should be (all_images_size, 3, 176, 176)
    print(f'--delete-- x.shape {x.shape}')

    # load to device
    if device.type == 'cuda':
        x = x.to(device=device)  
        y = y.float()
        y = y.to(device=device)
    
    '''
    feed data to model to get K dim result
    '''
    y_pred = model(x)
    print(f'--delete-- y_pred.shape {y_pred.shape}')
    
    '''
    restore dimension to y_pred by using the decoder
    '''
    # restore dimension to y_pred by using the decoder (from our pre-trained autoencoder)
    M_pred = dataset.autoEncoder.decodeWrapper(y_pred) # TODO: might need to put .to(device) here
    M_pred = M_pred.cpu().detach().numpy()
    # get M_truth
    M_truth = dataset.matrix_dataframe.to_numpy()  # this is the full 33538 dimensional vector (or 23073~ after reduction) from the original dataframe
    # assert equal sizes
    assert M_pred.shape == M_truth.shape
    # delete vectors used from the GPU
    del x
    del y 
    #
    return M_truth, M_pred
    





    return result


def get_model_by_name(name, dataset, hyperparams):
    '''
    prep:
    '''
    x0, y0, _ = dataset[0]  # NOTE that the third argument recieved here is "column" and is not currently needed
    in_size = x0.shape # note: if we need for some reason to add batch dimension to the image (from [3,176,176] to [1,3,176,176]) use x0 = x0.unsqueeze(0)  # ".to(device)"
    output_size = 1 if isinstance(y0, int) or isinstance(y0, float) else y0.shape[0] # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)  !!! #TODO: might need to be changed to a single number using .item() ... or .squeeze().item()
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
        output_size = 1 if isinstance(y0, int) or isinstance(y0, float) else y0.shape[0] # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)  !!! #TODO: might need to be changed to a single number using .item() ... or .squeeze().item()
                                                                # TODO: check if the output size is for one image, or for a batch !!! for now it is treated as for a single image
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

        # update the existing models laster layer
        input_size = model.fc.in_features
        output_size = 1 if isinstance(y0, int) or isinstance(y0, float) else y0.shape[0] # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)  !!! #TODO: might need to be changed to a single number using .item() ... or .squeeze().item()
                                                                # TODO: check if the output size is for one image, or for a batch !!! for now it is treated as for a single image
        model.fc = torch.nn.Linear(input_size, output_size, bias=True)
        model.fc.weight.data.zero_()
        model.fc.bias.data.zero_()
        return model
