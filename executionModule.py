import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
from sklearn.decomposition import NMF
from deepNetworkArchitechture import ConvNet


def runTest1_singleGenePrediction(dataset : Dataset, device):
    
    print("\n----- entered function runTest1 -----")


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
    im_size = ds_train[0][0].shape

    print(f'verify size of ds_train {len(ds_train)}')
    print(f'verify size of ds_test {len(ds_test)}')
    print(f'verify size of dl_train {len(dl_train)}')
    print(f'verify size of dl_test {len(dl_test)}')
    print(f'verify im_size {im_size}')
    print(f'verify size of dl_test {len(dl_test)}')
    print(f'verify batch_size is {batch_size} ')

    '''
    prepare model, loss and optimizer instances
    '''

    max_batches = 1000000 # ?
    x0, _ = dataset[0]
    print(f'A single image\'s shape will be like x0.shape : {x0.shape}')
    # add batch dimension
    x0 = x0.unsqueeze(0)  # ".to(device)"
    print(f'A single image\'s shape will be like x0.shape - after unsqueeze : {x0.shape}')
    in_size = x0.shape[1:]  # save it as only 3 parameters out of [1,3,176,176] - save as [3,176,176]
    output_size = 1  #notes on this line:
        # formerly known as - "out_classes". now, this will be the regression value FOR EACH SINGLE IMAGE (or so i think) #TODO: verify
        # TODO: note that i did not yet perform in softmax or any such thing
    channels = [32]  # these are the kernels if i remember correctly
    hidden_dims = [100]
    pool_every = 9999  # because of the parametes above, this practically means never ...
    print(f'verify in_size {in_size}')
    print(f'note - number of convolutions is supposed to be {len(channels)}')
    print(f'note - number of (hidden) linear layers is supposed to be {len(hidden_dims)}')
    model = ConvNet(in_size, output_size, channels=channels, pool_every=pool_every, hidden_dims=hidden_dims)
    if device.type == 'cuda':
        model = model.to(device=device)  # 030920 test: added cuda
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    '''
    now we can perform the test !
    '''


    # training loop - fourth try

    print("****** begin training ******")
    num_of_epochs = 10
    max_alowed_number_of_batches = 30  # the purpose of this var is if i dont realy want all of the batches to be trained uppon ... 
    # and so if this number is higher than the real number of batches - it means i will use all of the batchs for my traiining process
    # note that there are currently (030920) 120 batches - 120 batches * 25 images in each batch = 3000 images in ds_train
    num_of_batches = (len(ds_train) // batch_size)  # TODO: check this line
    if num_of_batches > max_alowed_number_of_batches:
        num_of_batches = max_alowed_number_of_batches

    # initialize variables
    total_correct_predictions_all_epochs = 0

    # note 2 loops here: external and internal
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
        print(f'               number of correct predictions: {num_of_correct_predictions_this_epoch} / {batch_size*num_of_batches}')
        total_correct_predictions_all_epochs += num_of_correct_predictions_this_epoch

        # todo: do we want to keep the average loss from this epoch ?
        unused_var = np.mean(loss_values_list)

        
    print(f'finished all epochs ! \nnum of total correct predicions: {total_correct_predictions_all_epochs} / {len(ds_train)}')

    print(f' FINISHED TRAINING ')

    print("\n----- finished function runTest1 -----\n")

    pass


def runTest2_allGenePrediction_dimReduction_KHighestVariances(dataset : Dataset, device):
    
    print("\n----- entered function runTest2_allGenePrediction_dimReduction_KHighestVariances -----")



    print("\n----- finished function runTest2_allGenePrediction_dimReduction_KHighestVariances -----\n")

    pass


def runTest3_allGenePrediction_dimReduction_NMF(dataset : Dataset, device):
    
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


def runTest4_allGenePrediction_dimReduction_AutoEncoder(dataset : Dataset, device):

    print("\n----- entered function runTest4_allGenePrediction_dimReduction_AutoEncoder -----")

    
    

    print("\n----- finished function runTest4_allGenePrediction_dimReduction_AutoEncoder -----\n")

    pass