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
            x, y, _ = data  # note :  x.shape is: torch.Size([25, 3, 176, 176]) y.shape is: torch.Size([25]) because the batch size is 25. 
                            # NOTE that the third argument recieved here is "column" and is not currently needed
            
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

            # checking for same values in the predcition y_pred as in ground truth y:
            # first, flatten the vectors, and ROUND THE VALUES ! NOTE !!!!!
            y_pred_rounded_flattened = torch.round(y_pred).type(torch.int).flatten()  #TODO: note this rounding process to the closest iteger !
            y_rounded_flattened = torch.round(y).type(torch.int).flatten()  #TODO: note this rounding process to the closest iteger !
            # print(f'--delete-- y.shape {y.shape}, y_pred.shape {y_pred.shape}, y_pred_rounded.shape {y_pred_rounded.shape}')
            num_of_correct_predictions_this_batch = torch.eq(y_rounded_flattened, y_pred_rounded_flattened).sum()
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

            # delete unneeded tesnors from GPU to save space #TODO: need to check that this code works
            del x, y, y_pred

        #end of inner loop
        print(f'\nfinished inner loop.\n')


        # data prints on the epoch that ended
        print(f'in this epoch: min loss {np.min(loss_values_list)} max loss {np.max(loss_values_list)}')
        print(f'               average loss {np.mean(loss_values_list)}')
        print(f'               number of correct predictions: {num_of_correct_predictions_this_epoch} / {dl_train.batch_size*num_of_batches}')

 
    print(f'finished all epochs !')
    print(" \ * / FINISHED train_prediction_model \ * / ")
    return model


def runExperimentWithModel_BasicConvNet(dataset : Dataset, hyperparams, device, dataset_name):
    '''
    hyperparams is a dict that should hold the following variables:
    batch_size, max_alowed_number_of_batches, precent_of_dataset_allocated_for_training, learning_rate, num_of_epochs,   
    channels, num_of_convolution_layers, hidden_dims, num_of_hidden_layers, pool_every
    '''
    print("\n----- entered function runExperimentWithModel_BasicConvNet -----")
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
    x0, y0, _ = dataset[0]  # NOTE that the third argument recieved here is "column" and is not currently needed
    in_size = x0.shape # note: if we need for some reason to add batch dimension to the image (from [3,176,176] to [1,3,176,176]) use x0 = x0.unsqueeze(0)  # ".to(device)"
    output_size = 1 if isinstance(y0, int) else y0.shape[0] # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)  !!! #TODO: might need to be changed to a single number using .item() ... or .squeeze().item()
    
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
    '''
    perform  an experiment to check the dimensionality restoration
    '''
    if dataset_name == 'NMF':
        runDimensionalityRestorationExperiment_with_NMF_DS(dataset=dataset, model=model, device=device)
        pass
    elif dataset_name == 'AE':
        runDimensionalityRestorationExperiment_with_AE_DS(dataset=dataset, model=model, device=device)
        pass
    else:
        # if we get here, this is not NMF or AE, and do nothing for now
        pass

    print("\n----- finished function runExperimentWithModel_BasicConvNet -----")
    
    #temporary ?
    # return model  #TODO: have to decide if the model should be returned or not .... if not return: NOTE: delete the model from GPU !!
    # delete unneeded tesnors from GPU to save space #TODO: need to check that this code works
    del model
    pass
    

def runDimensionalityRestorationExperiment_with_NMF_DS(dataset : Dataset, model, device):
    '''
    '''
    print("\n----- entered function runDimensionalityRestorationExperiment_with_NMF_DS -----")

    # # Verification of W,H matrices
    # print("Verification of W,H matrices:")  
    # print(f'W len {len(dataset.W)}')
    # print(f'W type {type(dataset.W)}')
    # print(f'W shape {dataset.W.shape}')
    # print(f'H len {len(dataset.H)}')
    # print(f'H len {type(dataset.H)}')
    # print(f'H shape {dataset.H.shape}')
    # print(f' and so: W*H = {dataset.W.shape}*{dataset.H.shape} should give us the right format')

    '''
    # take an image, run it through the model, restore its dimensions, and compare it to the original vecotr from the dataset
    '''
    x0, y0, column = dataset[0]
    x0 = x0.unsqueeze(0).to(device=device)  # add batch dimension (1,...) at beginning of the tensor's shape
    y0 = y0.unsqueeze(0).to(device=device)  # add batch dimension (1,...) at beginning of the tensor's shape
    y0_pred = model(x0)  # NOTE: assumption: model was already uploaded to cuda when it was trained
    
    '''
    restore dimension to y0_pred by performing: res =      W * y_pred 
                                                 (33538 x K) * (K * num_of_images)
    '''
    # both vecotors need a little preparation for the multiplication
    y0_pred_prepared = y0_pred.t().to(device=device) #note the transpose here !
    W_prepared = torch.from_numpy(dataset.W).float().to(device=device)

    # print(f'--delete-- verify: W_prepared.shape {W_prepared.shape}, y0_pred_prepared.shape {y0_pred_prepared.shape}')

    y0_pred_all_dims = torch.mm(W_prepared, y0_pred_prepared)  

    # print(f'--delete-- verify: x0.shape {x0.shape}, y0.shape {y0.shape}, y0_pred.shape {y0_pred.shape}, y0_pred_all_dims.shape {y0_pred_all_dims.shape}')

    y0_ground_truth_all_dims = dataset.matrix_dataframe.iloc[:, column].to_numpy()  # this is the full 33538 dimensional vector (or 23073~ after reduction) from the original dataframe

    # # trick to get num of identical elements between 2 NUMPY arrays 
    # # NOTE: the cuda tesnor needs a little conversion first from cuda to cpu and to the right dimension; and both vectors needs rounding to closest int (using np.rint)
    y0_groundtruth_prepared = np.rint(y0_ground_truth_all_dims)
    y0_pred_prepared = np.rint(y0_pred_all_dims.squeeze().cpu().detach().numpy())   # np.squeeze(np.rint(y0_pred_all_dims.cpu().detach().numpy()))

    # print(f'--delete-- verify:  y0_pred_prepared shape {y0_pred_prepared.shape} \ny0_groundtruth_prepared shape {y0_groundtruth_prepared.shape}')
    # print(f'--delete-- verify:  y0_pred_prepared {y0_pred_prepared} \ny0_groundtruth_prepared shape {y0_groundtruth_prepared}')

    # assert y0_groundtruth_prepared == y0_pred_prepared.shape  # check if needed
    number_of_identical_elements = np.sum(y0_groundtruth_prepared == y0_pred_prepared) 
                                                                                                      # note that "y0_pred_all_dims" needs to be copied from cuda to cpu in order to use numpy                                                                                                  

    print(f'corret prediction after dimensionality restoration: {number_of_identical_elements} out of {len(y0_ground_truth_all_dims)}')


    print("\n----- finished function runDimensionalityRestorationExperiment_with_NMF_DS -----\n")

    pass


def runDimensionalityRestorationExperiment_with_AE_DS(dataset : Dataset, model, device):
    '''
    '''
    print("\n----- entered function runDimensionalityRestorationExperiment_with_AE_DS -----")

    '''
    # take an image, run it through the model, restore its dimensions, and compare it to the original vecotr from the dataset
    '''
    # note that since this is a small test over 1 image, i dont copy to cuda, and i need to unsqueeze x0 and y0 to get their batch dim
    x0, y0, column = dataset[0]
    x0 = x0.unsqueeze(0).to(device=device)  # add batch dimension (1,...) at beginning of the tensor's shape
    y0 = y0.unsqueeze(0).to(device=device)  # add batch dimension (1,...) at beginning of the tensor's shape
    y0_pred = model(x0)  # NOTE: assumption: model was already uploaded to cuda when it was trained
    # restore dimension to y0_pred by using the decoder (from our pre-trained autoencoder)
    y0_pred_all_dims = dataset.autoEncoder.decodeWrapper(y0_pred)

    # print(f'--delete-- verify: x0.shape {x0.shape}, y0.shape {y0.shape}, y0_pred.shape {y0_pred.shape}, y0_pred_all_dims.shape {y0_pred_all_dims.shape}')

    y0_ground_truth_all_dims = dataset.matrix_dataframe.iloc[:, column].to_numpy()  # this is the full 33538 dimensional vector (or 23000~ after reduction) from the original dataframe

    # # trick to get num of identical elements between 2 NUMPY arrays 
    # # NOTE: the cuda tesnor needs a little conversion first from cuda to cpu and to the right dimension; and both vectors needs rounding to closest int (using np.rint)
    y0_groundtruth_prepared = np.rint(y0_ground_truth_all_dims)
    y0_pred_prepared = np.rint(y0_pred_all_dims.squeeze().cpu().detach().numpy())   # np.squeeze(np.rint(y0_pred_all_dims.cpu().detach().numpy()))

    # print(f'--delete-- verify:  y0_pred_prepared shape {y0_pred_prepared.shape} \ny0_groundtruth_prepared shape {y0_groundtruth_prepared.shape}')
    # print(f'--delete-- verify:  y0_pred_prepared {y0_pred_prepared} \ny0_groundtruth_prepared shape {y0_groundtruth_prepared}')

    # assert y0_groundtruth_prepared == y0_pred_prepared.shape  # check if needed
    number_of_identical_elements = np.sum(y0_groundtruth_prepared == y0_pred_prepared) 
                                                                                                      # note that "y0_pred_all_dims" needs to be copied from cuda to cpu in order to use numpy                                                                                                  

    print(f'corret prediction after dimensionality restoration: {number_of_identical_elements} out of {len(y0_ground_truth_all_dims)}')

    print("\n----- finished function runDimensionalityRestorationExperiment_with_AE_DS -----\n")

    pass


def runExperimentWithModel_STNet_DenseNet121(dataset : Dataset, hyperparams, device, dataset_name):
    '''
    hyperparams is a dict that should hold the following variables:
    batch_size, max_alowed_number_of_batches, precent_of_dataset_allocated_for_training, learning_rate, num_of_epochs,   
    channels, num_of_convolution_layers, hidden_dims, num_of_hidden_layers, pool_every
    '''
    print("\n----- entered function runExperimentWithModel_BasicConvNet -----")
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
    x0, y0, _ = dataset[0]  # NOTE that the third argument recieved here is "column" and is not currently needed
    in_size = x0.shape # note: if we need for some reason to add batch dimension to the image (from [3,176,176] to [1,3,176,176]) use x0 = x0.unsqueeze(0)  # ".to(device)"
    output_size = 1 if isinstance(y0, int) else y0.shape[0] # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)  !!! #TODO: might need to be changed to a single number using .item() ... or .squeeze().item()
    
    # create the models
    # explanation of all models in: https://pytorch.org/docs/stable/torchvision/models.html
    model = torchvision.models.densenet121(pretrained=False) # TODO: not sure if this should be true or false ...
    stnet.utils.nn.set_out_features(model, outputs) # performs operation inplace #TODO: what is the "outputs" ?
    '''
    original code from github to determine the outputs above:
    # Find number of required outputs
        if args.task == "tumor":
            outputs = 2
        elif args.task == "gene":
            outputs = train_dataset[0][2].shape[0]
        elif args.task == "geneb":
            outputs = 2 * train_dataset[0][2].shape[0]
        elif args.task == "count":
            outputs = 1
    '''

    # create the loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], momentum=hyperparams['momentum']) #TODO: add momentum to hyperparams dict in the notebook.
                                                                                                                    #TODO: in the paper momentum is 0.9 and lr is 1e-6 (1x10^-6)

    '''
    now we can perform the test !
    '''

    # TODO: NOTE: actuall training loop from the code of the original paper can be found here:
    # https://github.com/bryanhe/ST-Net/blob/master/stnet/cmd/run_spatial.py
    # starting from line 243

    train_prediction_model(model_to_train=model, ds_train=ds_train, dl_train=dl_train, loss_fn=loss_fn, 
                                   optimizer=optimizer, num_of_epochs_wanted=hyperparams['num_of_epochs'], 
                                   max_alowed_number_of_batches=hyperparams['max_alowed_number_of_batches'],
                                   device=device)
                                   

    # TODO: test the model ?
    '''
    perform  an experiment to check the dimensionality restoration
    '''
    if dataset_name == 'NMF':
        runDimensionalityRestorationExperiment_with_NMF_DS(dataset=dataset, model=model, device=device)
        pass
    elif dataset_name == 'AE':
        runDimensionalityRestorationExperiment_with_AE_DS(dataset=dataset, model=model, device=device)
        pass
    else:
        # if we get here, this is not NMF or AE, and do nothing for now
        pass

    print("\n----- finished function runExperimentWithModel_BasicConvNet -----")
    
    #temporary ?
    # return model  #TODO: have to decide if the model should be returned or not .... if not return: NOTE: delete the model from GPU !!
    # delete unneeded tesnors from GPU to save space #TODO: need to check that this code works
    del model
    pass


def runExperimentWithModel_PreTrainedNets(dataset : Dataset, hyperparams, device, dataset_name):
    '''
    hyperparams is a dict that should hold the following variables:
    batch_size, max_alowed_number_of_batches, precent_of_dataset_allocated_for_training, learning_rate, num_of_epochs,   
    channels, num_of_convolution_layers, hidden_dims, num_of_hidden_layers, pool_every
    '''
    print("\n----- entered function runExperimentWithModel_BasicConvNet -----")
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
    x0, y0, _ = dataset[0]  # NOTE that the third argument recieved here is "column" and is not currently needed
    in_size = x0.shape # note: if we need for some reason to add batch dimension to the image (from [3,176,176] to [1,3,176,176]) use x0 = x0.unsqueeze(0)  # ".to(device)"
    output_size = 1 if isinstance(y0, int) else y0.shape[0] # NOTE: if y0 is an int, than the size of the y0 tensor is 1. else, its size is K (K == y0.shape)  !!! #TODO: might need to be changed to a single number using .item() ... or .squeeze().item()
    
    # create the models
    # explanation of all models in: https://pytorch.org/docs/stable/torchvision/models.html
    model_vgg_11 = torchvision.models.vgg11(pretrained=True)
    model_vgg_11_bn = torchvision.models.vgg11_bn(pretrained=True) # with batch normalization (bn)
    model_vgg_16 = torchvision.models.vgg16(pretrained=True)
    model_inception_v3 = torchvision.models.inception_v3(pretrained=True)
    model_densenet_121 = torchvision.models.densenet121(pretrained=True) 
    model_densenet_161 = torchvision.models.densenet161(pretrained=True) 
    model_list_full = [model_vgg_11, model_vgg_11_bn, model_vgg_16, model_inception_v3, model_densenet_121, model_densenet_161]


    # create the loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    '''
    now we can perform the test !
    '''


    # TODO: test the models ?


    print("\n----- finished function runExperimentWithModel_BasicConvNet -----")
    
    pass


def set_out_features_for_pretrained_models(model, outputs):
    """Changes number of outputs for the model.
    The change occurs in-place, but the new model is also returned.

    Note: this function was taken from the original paper as is.
    """

    if (isinstance(model, torchvision.models.AlexNet) or
        isinstance(model, torchvision.models.VGG)):
        inputs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(inputs, outputs, bias=True)
        model.classifier[-1].weight.data.zero_()
        model.classifier[-1].bias.data.zero_()
    elif (isinstance(model, torchvision.models.ResNet) or 
          isinstance(model, torchvision.models.Inception3)):
        inputs = model.fc.in_features
        model.fc = torch.nn.Linear(inputs, outputs, bias=True)
        model.fc.weight.data.zero_()
        model.fc.bias.data.zero_()
    elif isinstance(model, torchvision.models.DenseNet):
        inputs = model.classifier.in_features
        model.classifier = torch.nn.Linear(inputs, outputs, bias=True)
        model.classifier.weight.data.zero_()
        model.classifier.bias.data.zero_()
    else:
        raise NotImplementedError()

    return model