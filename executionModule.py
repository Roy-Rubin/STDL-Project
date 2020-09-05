import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np

from deepNetworkArchitechture import ConvNet


def runTest1(dataset : Dataset, device):

    print("\n----- entered function runTest1 -----")

    '''
    prep our dataset and dataloaders
    '''
    batch_size = 25
    split_lengths = [int(len(dataset) * 0.9), int(len(dataset) * 0.1)]  # this didnt work .... bad numbers \o/
    split_lengths = [3000, 813]  # since there are 3813 samples overall, 3000 is about 78% of the ds
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
    output_size = batch_size  # formerly known as - "out_classes". now, this will be the regression value # todo: note that i did not yet perform in softmax or any such thing
    channels = [32]  # these are the kernels if i remember correctly
    hidden_dims = [100]
    pool_every = 9999  # because of the parametes above, this practically means never ...
    print(f'verify in_size {in_size}')
    print(f'note - number of convolutions is supposed to be {len(channels)}')
    print(f'note - number of (hidden) linear layers is supposed to be {len(hidden_dims)}')
    model = ConvNet(in_size, output_size, channels=channels, pool_every=pool_every,
                           hidden_dims=hidden_dims)
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    '''
    now we can perform the test !
    '''


    # training loop - fourth try

    print("****** begin training ******")
    num_of_epochs = 5
    max_number_of_batches = 10  # the purpose of this var is if i dont realy want all of the batches to be trained uppon
    num_of_batches = (len(ds_train) // batch_size)  # TODO: check this line
    if num_of_batches > max_number_of_batches:
        num_of_batches = max_number_of_batches

    # note 2 loops here.
    for iteration in range(num_of_epochs):
        print(f'iteration {iteration+1} of {num_of_epochs} epochs')
        dl_iter = iter(dl_train)  # iterator over the dataloader. called only once, outside of the loop.
        loss_values_list = []
        for batch_index in range(num_of_batches):
            print(f'batch {batch_index+1} of {num_of_batches} batches')
            # get current x batch
            data = next(dl_iter)  # note: data is a list with 2 elements:  data[0] is: <class 'torch.Tensor'> data[1] is: <class 'torch.Tensor'>
            x, y = data  # note :  x.shape is: torch.Size([25, 3, 176, 176]) y.shape is: torch.Size([25]) because the batch size is 25
            
            # TODO: check if this is needed
            #x.to(device=device)
            #y.to(device=device)
            
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)  # TODO: check if .to(device=device) is needed

            # Compute (and print) loss.
            loss = loss_fn(y_pred, y)  # todo: check oder
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
        print(f'finished inner loop.')
        print("loss_values_list len: ", len(loss_values_list))
        print(f'loss_values_list is {loss_values_list}')
        print(f'average loss for this epoch is {np.mean(loss_values_list)}')
        print(f'min loss for this epoch is {np.min(loss_values_list)} max loss for this epoch is {np.max(loss_values_list)}')

        # todo: do we want to keep the average loss from this epoch ?
        unused_var = np.mean(loss_values_list)


    print("\n----- finished function runTest1 -----\n")

    pass