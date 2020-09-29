import numpy as np

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
