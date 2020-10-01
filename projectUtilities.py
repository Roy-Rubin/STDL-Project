import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


def compare_matrices(M_truth, M_pred, Baseline=None): #note the None if not needed
    # TODO: might need to move to utilities
    '''
    method - calculate distance between matrices
    '''
    if Baseline is None:
        print(f'recieved Baseline=None. errors with it will be 0')
    error1 = calculate_distance_between_matrices(M_truth, M_pred)
    error2 = calculate_distance_between_matrices(M_truth, Baseline)
    error3 = calculate_distance_between_matrices(M_pred, Baseline)
    print(f'distance between M_truth, M_pred: {error1}')
    print(f'distance between M_truth, Baseline: {error2}')
    print(f'distance between M_pred, Baseline: {error3}')

    pass


def calculate_distance_between_matrices(matrix1,matrix2):
    '''
    step 1: check (and convert if needed) that the matrices are of numpy ndarray type
    step 2: check distance using FROBENIUS NORM
    '''
    # step 1
    m1, m2 = matrix1, matrix2
    if m1 is None or m2 is None:
        return 0
    if not isinstance(m1, np.ndarray):
        # assumption: it means that it is a pandas object
        m1 = matrix1.to_numpy()
    if not isinstance(m2, np.ndarray):
        # assumption: it means that it is a pandas object
        m2 = matrix1.to_numpy()
    assert m1.shape == m2.shape
    '''
    NOTE: !
    np.linalg.norm(var)  
    when no order is given to the method above, there are 2 cases:
    if var is a vector, then calculate 2-norm
    if var is a matrix, then calculate frobenius-norm
    '''
    temp = m1-m2
    distance = np.linalg.norm(temp)  
    return distance


def get_variance_of_gene(gene_name, matrix_df, row_mapping, features_df):
    
    '''
    get the variance of a specific gene over all the samples
    '''
    gene_index_in_old_df = features_df.index[features_df['gene_names'] == gene_name].item()  # the old df is th original one before preprocessing
    # new indices
    row = row_mapping.index[row_mapping['original_index_from_matrix_dataframe'] == gene_index_in_old_df].item() # assumption: only one item is returned
    # variance values
    temp = pd.DataFrame(matrix_df.iloc[row,:])
    gene_variance_value = temp.var()
    return gene_variance_value.item()


def printInfoAboutDataset(dataset):
    print(f'printing information about the dataset:')
    print(f'size of the dataset (==number of images in the image folder) {dataset.size_of_dataset}')
    print(f'num_of_samples_matrix_df in the dataset (==number of columns in matrix_dataframe) {dataset.num_of_samples_matrix_df}')
    print(f'num_of_features_matrix_df in the dataset (==number of rows in matrix_dataframe) {dataset.num_of_features_matrix_df}')
    

def printInfoAboutDFs(matrix_dataframe, features_dataframe, barcodes_datafame):
    print("\nprint data regarding the dataframes:")
    print("\nfeatures_dataframe:")
    print(features_dataframe.info())
    print(features_dataframe.head(5))
    print("\nbarcodes_datafame:")
    print(barcodes_datafame.info())
    print(barcodes_datafame.head(5))
    print("\nmatrix_dataframe:")
    print(matrix_dataframe.info())
    print(matrix_dataframe.head(5))
    # note that matrix_dataframe.min() gives you the min value in each column. if you want the entire min, it should be:
    # df.min().min()   or   df.to_numpy().min()
    print(
        f'\nmin value in matrix_dataframe {matrix_dataframe.min().min()} max value in matrix_dataframe {matrix_dataframe.max().max()}')
    import numpy as np
    print(
        f'\nmedian value in matrix_dataframe {np.median(matrix_dataframe.values)} mean value in matrix_dataframe {np.mean(matrix_dataframe.values)}')

    list_of_lists_from_df = matrix_dataframe.values.tolist()
    import itertools
    one_big_list_of_values_from_matrix_df = list(itertools.chain.from_iterable(list_of_lists_from_df))
    number_of_different_values = len(set(one_big_list_of_values_from_matrix_df))
    print(
        f'\nnumber of different values in matrix_dataframe is  {number_of_different_values} ')
    
    num_of_values_in_matrix = len(matrix_dataframe.index)*len(matrix_dataframe.columns) # the first one is the num of rows    

    from collections import Counter
    print(
        f'\nlist of 10 most common values in matrix_dataframe is: ')
    for index, list_item in enumerate(Counter(one_big_list_of_values_from_matrix_df).most_common(10)):
        value, num_of_apearences = list_item
        print(f'{index+1}: the value {value} appeared {num_of_apearences} times (constitutes {(num_of_apearences/num_of_values_in_matrix)*100:.5f}% of the matrix values)') 
        # "":.5f" rounds to 00.00000

    pass


def printInfoAboutReducedDF(matrix_dataframe):
    print("\nprint data regarding the reduced dataframe:")

    print(matrix_dataframe.info())
    print(matrix_dataframe.head(5))
    # note that matrix_dataframe.min() gives you the min value in each column. if you want the entire min, it should be:
    # df.min().min()   or   df.to_numpy().min()
    print(
        f'\nmin value in matrix_dataframe {matrix_dataframe.min().min()} max value in matrix_dataframe {matrix_dataframe.max().max()}')
    import numpy as np
    print(
        f'\nmedian value in matrix_dataframe {np.median(matrix_dataframe.values)} mean value in matrix_dataframe {np.mean(matrix_dataframe.values)}')

    list_of_lists_from_df = matrix_dataframe.values.tolist()
    import itertools
    one_big_list_of_values_from_matrix_df = list(itertools.chain.from_iterable(list_of_lists_from_df))
    number_of_different_values = len(set(one_big_list_of_values_from_matrix_df))
    print(
        f'\nnumber of different values in matrix_dataframe is  {number_of_different_values} ')
    
    num_of_values_in_matrix = len(matrix_dataframe.index)*len(matrix_dataframe.columns) # the first one is the num of rows
    from collections import Counter
    print(
        f'\nlist of 10 most common values in matrix_dataframe is: ')
    for index, list_item in enumerate(Counter(one_big_list_of_values_from_matrix_df).most_common(10)):
        value, num_of_apearences = list_item
        print(f'{index+1}: the value {value} appeared {num_of_apearences} times (constitutes {(num_of_apearences/num_of_values_in_matrix)*100:.5f}% of the matrix values)') 
        # "":.5f" rounds to 00.00000

    pass


def printInfoAboutImageFolderDataset(dataset_object):
    print(f'\ndataset loaded. found {len(dataset_object)} images in dataset folder.')
    print(f'returned object type: {type(dataset_object)}')

    print(f'ImageFolder\'s root == root directory: {dataset_object.root}')
    print(f'ImageFolder\'s classes len == number of sub folders with images: {len(dataset_object.classes)}')
    print(f'ImageFolder\'s classes == all classes names == all subfolder names: {dataset_object.classes}')
    print(
        f'ImageFolder\'s class_to_idx == map from class (subfolder) index to class (subfolder) name: {dataset_object.class_to_idx}')
    ## NOTE: apparently, since we did not supply a Loader, then dataset_object.samples and dataset_object.imgs behave the same
    print(
        f'ImageFolder\'s imgs[0] == first image: {dataset_object.imgs[0]}  <-- note that the class is currently not relevant')
    print(
        f'ImageFolder\'s samples[0] == first image: {dataset_object.samples[0]}  <-- note that the class is currently not relevant')
    print(
        f'ImageFolder: asserting that samples len {len(dataset_object.samples)} == imgs len {len(dataset_object.imgs)}')
    assert (len(dataset_object.samples) == len(dataset_object.imgs))
    print(
        f'ImageFolder[0] == __getitem__ method: note that this is a 2d tuple of a tensor and a y_value: \n{dataset_object[0]} <-- note that the class is currently not relevant')

    pass


def printInfoAboutCustomConcatanatedImageFolderDataset(dataset_object):
    print(f'\nConcatanated dataset loaded. found {len(dataset_object)} images in dataset folder.')
    print(f'returned object type: {type(dataset_object)}')
    print(f'dataset_lengths_list: {dataset_object.dataset_lengths_list}')
    print(f'index_offsets: {dataset_object.index_offsets}')
    print(f'list_of_image_filenames len: {len(dataset_object.list_of_image_filenames)}')
    print(f'list_of_image_filenames first few name: {dataset_object.list_of_image_filenames[0:5]}')

    pass


def plot_Single_Gene_PredAndTrue(dataset, M_pred, M_truth, model_name, dataset_name):
    
    print("\n----- entered function plot_Single_Gene_PredAndTrue -----")

    '''
    Plot data to compare matrices
    '''
    plt.clf()  # clears previous plots
    # create a scatter
    plt.scatter(x=M_truth, y=M_pred, label='M_truth VS M_pred')
    # create a line
    lower_x_bound = 0 # lower_x_bound = np.min(M_truth) - 0.1
    upper_x_bound = np.max(M_pred) + 1  # upper_x_bound = np.max(M_truth) + 1
    num_of_dots_in_line = 100
    x = np.linspace(lower_x_bound,upper_x_bound,num_of_dots_in_line) # linspace() function to create evenly-spaced points in a given interval
    y = x  # to plot y=x we'll create a y variable that is exactly like x
    plt.plot(x, y, '--k', label='y=x plot') # create a line # "--k" means black dashed line
    # set surroundings
    plt.xlabel(f'M_truth values')
    plt.ylabel(f'M_pred values')
    plt.title(f'Result of comparison between M_truth VS M_pred\nSingle Gene experiment with model: {model_name}\ngene chosen: {dataset.gene_name}')
    plt.legend()
    # filename = f'{dataset_name}_{model_name}_comparison'
    # plt.savefig(f'{filename}.png', bbox_inches='tight')
    plt.show()
    plt.clf()


    # testing !!!

    M_truth_upscaled = [np.expm1(val) for val in list(M_truth)]
    M_pred_upscaled = [np.expm1(val) for val in list(M_pred)]
    # create a scatter
    plt.scatter(x=M_truth_upscaled, y=M_pred_upscaled, label='M_truth VS M_pred')
    # create a line
    lower_x_bound = 0 # lower_x_bound = np.min(M_truth) - 0.1
    upper_x_bound = np.max(M_pred_upscaled) + 1  # upper_x_bound = np.max(M_truth) + 1
    num_of_dots_in_line = 100
    x = np.linspace(lower_x_bound,upper_x_bound,num_of_dots_in_line) # linspace() function to create evenly-spaced points in a given interval
    y = x  # to plot y=x we'll create a y variable that is exactly like x
    plt.plot(x, y, '--k', label='y=x plot') # create a line # "--k" means black dashed line
    # set surroundings
    plt.xlabel(f'M_truth values')
    plt.ylabel(f'M_pred values')
    plt.title(f'Result of comparison between M_truth VS M_pred\nSingle Gene experiment with model: {model_name}\ngene chosen: {dataset.gene_name}')
    plt.legend()
    plt.show()
    plt.clf()
    
    '''
    Plot data to compare with the large biopsy image
    '''

    '''
    First, gather info
    '''
    ### create the value column in the dataframe  ###
    list_of_values_true = []
    list_of_values_pred = []
    x_list = []
    y_list = []

    for index in range(dataset.num_of_images_with_no_augmentation):
        # get file's name
        if hasattr(dataset.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            curr_filename = dataset.imageFolder.samples[index][0]
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            _, curr_filename = dataset.imageFolder[index]

        # get the sample's name from its absolute path and file name
        curr_sample_name = curr_filename.partition('_')[0].partition('/images/')[2]  # first partition to get everything before the first _ , second partition to get everything after /images/

        # get the y value's COLUMN in the gene expression matrix df (with help from the barcodes df)
        index_in_barcoes_df = dataset.barcodes_dataframe.index[dataset.barcodes_dataframe['barcodes'] == curr_sample_name].item() # assumption: only 1 item is returned
        column = dataset.column_mapping.index[dataset.column_mapping['original_index_from_matrix_dataframe'] == index_in_barcoes_df].item() # assumption: only one item is returned
        
        # append the gene_exp value (pred or true) for that barcode
        list_of_values_true.append(M_truth[column])
        list_of_values_pred.append(M_pred[column])

        # get the x and y values for that image name
        x = curr_filename.partition('_')[2].partition('_')[0].partition('x')[2]
        y = curr_filename.partition('_')[2].partition('_')[2].partition('_')[0].partition('y')[2]
        
        # append x and y values. note the conversion - its because they are strings !
        x_list.append(int(x))
        y_list.append(int(y))  

    '''
    Make preparations for the plot
    '''
    # decrease sizes for plot reasons
    x_list = [x - min(x_list) for x in x_list]  # decreasing the size ...
    y_list = [x - min(y_list) for x in y_list]  # decreasing the size ...
    #
    x_boundry = int(max(x_list)) + 1
    y_boundry = int(max(y_list)) + 1
    # modulate up (~un-normalize the log1p normalization) because otherwise colors dont work
    list_of_values_true = [np.expm1(true_val)+1 for true_val in list_of_values_true]
    list_of_values_pred = [np.expm1(pred_val)+1 for pred_val in list_of_values_pred]
    
    # NOTE !!!! the low mid high values - will be built on TRUE VALUES but used also for PRED VALUES !!!!!
    list_sorted = sorted(list_of_values_true)
    n = len(list_sorted)
    low_val = list_sorted[int(2 * n/5)-1]
    mid_val = list_sorted[int(3 * n/5)-1]
    high_val = list_sorted[int(4 * n/5)-1]
    # create the (sparse de-facto but not sparse python-wise) matrices
    # init them empty
    fill_value = 0  
    low_T = np.full(shape=[x_boundry,y_boundry], fill_value=fill_value) # values is a 2d matrix - each entry is a color
    mid_T = np.full(shape=[x_boundry,y_boundry], fill_value=fill_value) # values is a 2d matrix - each entry is a color
    high_T = np.full(shape=[x_boundry,y_boundry], fill_value=fill_value) # values is a 2d matrix - each entry is a color
    very_high_T = np.full(shape=[x_boundry,y_boundry], fill_value=fill_value) # values is a 2d matrix - each entry is a color
    low_P = np.full(shape=[x_boundry,y_boundry], fill_value=fill_value) # values is a 2d matrix - each entry is a color
    mid_P = np.full(shape=[x_boundry,y_boundry], fill_value=fill_value) # values is a 2d matrix - each entry is a color
    high_P = np.full(shape=[x_boundry,y_boundry], fill_value=fill_value) # values is a 2d matrix - each entry is a color
    very_high_P = np.full(shape=[x_boundry,y_boundry], fill_value=fill_value) # values is a 2d matrix - each entry is a color

    # add values to the matrices
    index = 0
    for x, y, true_val, pred_val in zip(x_list, y_list, list_of_values_true, list_of_values_pred):
        index += 1
        # add to true matrices
        low_T[x,y] = true_val if true_val <= low_val else fill_value
        mid_T[x,y] = true_val  if true_val > low_val and true_val <= mid_val else fill_value
        high_T[x,y] = true_val  if true_val > mid_val and true_val <= high_val else fill_value
        very_high_T[x,y] = true_val  if true_val > high_val else fill_value
        # add to pred matrices
        low_P[x,y] = pred_val if pred_val <= low_val else fill_value
        mid_P[x,y] = pred_val  if pred_val > low_val and pred_val <= mid_val else fill_value
        high_P[x,y] = pred_val  if pred_val > mid_val and pred_val <= high_val else fill_value
        very_high_P[x,y] = pred_val  if pred_val > high_val else fill_value

    '''
    Plot !
    '''
    # plot figure for M_truth !!!
    plt.figure(figsize=(8,8))
    plt.spy(low_T, markersize=4, color='lime', label='Low Values')
    plt.spy(mid_T, markersize=4, color='yellow', label='Medium Values')
    plt.spy(high_T, markersize=4, color='deepskyblue', label='High Values')  # maybe 'violet' instead ?
    plt.spy(very_high_T, markersize=4, color='red', label='Very High Values')
    plt.legend()
    plt.xlabel(f'X coordinates')
    plt.ylabel(f'Y coordinates')
    plt.title(f'Plot of M_truth values\nSingle Gene experiment with model: {model_name}\ngene chosen: {dataset.gene_name}', fontsize=15)
    plt.show()
    plt.clf()

    # plot figure for M_pred !!!
    plt.figure(figsize=(8,8))
    plt.spy(low_P, markersize=4, color='lime', label='Low Values')
    plt.spy(mid_P, markersize=4, color='yellow', label='Medium Values')
    plt.spy(high_P, markersize=4, color='deepskyblue', label='High Values')  # maybe 'violet' instead ?
    plt.spy(very_high_P, markersize=4, color='red', label='Very High Values')
    plt.legend()
    plt.xlabel(f'X coordinates')
    plt.ylabel(f'Y coordinates')
    plt.title(f'Plot of M_pred values\nSingle Gene experiment with model: {model_name}\ngene chosen: {dataset.gene_name}', fontsize=15)
    plt.show()
    plt.clf()

    '''
    final note:
    options that didnt work here:
    scatter
    pcolor
    pcolormesh
    imshow
    '''
    pass
    print("\n----- finished function plot_Single_Gene_PredAndTrue -----")
