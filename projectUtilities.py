
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
    # df_min_value = matrix_dataframe.min().min()
    # df_max_value = matrix_dataframe.max().max()
    # print(
    #     f'\nmin value in matrix_dataframe {df_min_value} appears {list(matrix_dataframe.to_numpy().flatten).count(df_min_value)} times;')
    # print(
    #     f'max value in matrix_dataframe {df_max_value} appears {list(matrix_dataframe.to_numpy().flatten).count(df_max_value)} times')
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

    print(f'\nplotting a scatter plot of all values in matrix_dataframe (to see the distribution of values)')

    #TODO: do this : create a plot of all the data !
    # import matplotlib.pyplot as plt
    # plt.hist(one_big_list_of_values_from_matrix_df, density=False, bins=number_of_bins)  # `density=False` would make counts
    # plt.ylabel('what is y')
    # plt.xlabel('what is x')
    # plt.title('Histogram (allegedly)')
    # plt.show()
    # import seaborn as sns
    # # sns.distplot(one_big_list_of_values_from_matrix_df)
    # # plt.show()
    # sns.countplot(data=matrix_dataframe)
    # plt.show()

    # # testing "tolist()" property
    # feature_ids = features_dataframe['feature_ids'].tolist()
    # gene_names = features_dataframe['gene_names'].tolist()
    # feature_types = features_dataframe['feature_types'].tolist()
    # print()
    # print(f'feature_ids size {len(feature_ids)}')
    # print(f'feature_ids first 5 are: {feature_ids[0:5]}')
    # print(f'gene_names size {len(gene_names)}')
    # print(f'gene_names first 5 are: {gene_names[0:5]}')
    # print(f'feature_types size {len(feature_types)}')
    # print(f'feature_types first 5 are: {feature_types[0:5]}')
    # print(f'barcodes size {len(barcodes)}')
    # print(f'barcodes first 5 are: {barcodes[0:5]}')
    pass


def printInfoAboutReducedDF(matrix_dataframe):
    print("\nprint data regarding the reduced dataframe:")

    print(matrix_dataframe.info())
    print(matrix_dataframe.head(5))
    # note that matrix_dataframe.min() gives you the min value in each column. if you want the entire min, it should be:
    # df.min().min()   or   df.to_numpy().min()
    print(
        f'\nmin value in matrix_dataframe {matrix_dataframe.min().min()} max value in matrix_dataframe {matrix_dataframe.max().max()}')
    # df_min_value = matrix_dataframe.min().min()
    # df_max_value = matrix_dataframe.max().max()
    # print(
    #     f'\nmin value in matrix_dataframe {df_min_value} appears {list(matrix_dataframe.to_numpy().flatten).count(df_min_value)} times;')
    # print(
    #     f'max value in matrix_dataframe {df_max_value} appears {list(matrix_dataframe.to_numpy().flatten).count(df_max_value)} times')
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

    print(f'\nplotting a scatter plot of all values in matrix_dataframe (to see the distribution of values)')

    #TODO: do this : create a plot of all the data !
    # import matplotlib.pyplot as plt
    # plt.hist(one_big_list_of_values_from_matrix_df, density=False, bins=number_of_bins)  # `density=False` would make counts
    # plt.ylabel('what is y')
    # plt.xlabel('what is x')
    # plt.title('Histogram (allegedly)')
    # plt.show()
    # import seaborn as sns
    # # sns.distplot(one_big_list_of_values_from_matrix_df)
    # # plt.show()
    # sns.countplot(data=matrix_dataframe)
    # plt.show()

    # # testing "tolist()" property
    # feature_ids = features_dataframe['feature_ids'].tolist()
    # gene_names = features_dataframe['gene_names'].tolist()
    # feature_types = features_dataframe['feature_types'].tolist()
    # print()
    # print(f'feature_ids size {len(feature_ids)}')
    # print(f'feature_ids first 5 are: {feature_ids[0:5]}')
    # print(f'gene_names size {len(gene_names)}')
    # print(f'gene_names first 5 are: {gene_names[0:5]}')
    # print(f'feature_types size {len(feature_types)}')
    # print(f'feature_types first 5 are: {feature_types[0:5]}')
    # print(f'barcodes size {len(barcodes)}')
    # print(f'barcodes first 5 are: {barcodes[0:5]}')
    pass


def printInfoAboutCustomDataset(dataset_object):
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

    