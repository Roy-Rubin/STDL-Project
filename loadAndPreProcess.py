import csv
import gzip
import os

import matplotlib
import scipy.io
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as torchTransform
from torchvision.datasets import ImageFolder, DatasetFolder


def load_dataframes_from_mtx_and_tsv_new(path_to_mtx_tsv_files_dir):
    '''

    :param path:
    :return:


    original loading code from website:
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/matrices

    # matrix_dir = "..."
    # mat = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))
    #
    # features_path = os.path.join(matrix_dir, "features.tsv.gz")
    # feature_ids = [row[0] for row in csv.reader(gzip.open(features_path), delimiter="\t")]
    # gene_names = [row[1] for row in csv.reader(gzip.open(features_path), delimiter="\t")]
    # feature_types = [row[2] for row in csv.reader(gzip.open(features_path), delimiter="\t")]
    # barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
    # barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path), delimiter="\t")]

    '''
    print("\n----- entered function load_dataframes_from_mtx_and_tsv -----")

    # originaly:
    # matrix_dir = "..."
    # mat = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))
    # updated:
    # path_to_matrix = "C:/Users/royru/Downloads/spatialGeneExpression/matrix.mtx"  # TODO: note no gz
    path_to_matrix = path_to_mtx_tsv_files_dir + "/matrix.mtx"  # TODO: note no gz
    matrix = scipy.io.mmread(path_to_matrix)
    matrix_dataframe = pd.DataFrame.sparse.from_spmatrix(
        matrix)  # todo: note sure this works. from: https://pandas.pydata.org/docs/user_guide/sparse.html
    print("-DBG-: finished reading matrix.mtx")

    # path_to_features = "C:/Users/royru/Downloads/spatialGeneExpression/features.tsv"  # TODO: note no gz
    path_to_features = path_to_mtx_tsv_files_dir + "/features.tsv"  # TODO: note no gz
    # features_tsv_reader = csv.reader(path_to_features, delimiter="\t") # todo: delete this ?
    features_dataframe = pd.read_csv(path_to_features, sep='\t', header=None)
    features_dataframe.columns = ['feature_ids', 'gene_names', 'feature_types']  # giving columns their names
    print("-DBG-: finished reading features.tsv")

    # path_to_barcodes = "C:/Users/royru/Downloads/spatialGeneExpression/barcodes.tsv"  # TODO: note no gz
    path_to_barcodes = path_to_mtx_tsv_files_dir + "/barcodes.tsv"  # TODO: note no gz
    # barcodes_tsv_file = csv.reader(path_to_barcodes, delimiter="\t") # todo: delete this ?
    barcodes_datafame = pd.read_csv(path_to_barcodes, sep='\t', header=None)
    barcodes_datafame.columns = ['barcodes']  # giving columns their names
    print("-DBG-: finished reading barcodes.tsv")
    barcodes = barcodes_datafame['barcodes'].tolist()

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
    df_min_value = matrix_dataframe.min().min()
    df_max_value = matrix_dataframe.max().max()
    print(
        f'\nmin value in matrix_dataframe {df_min_value} appears {list(df.to_numpy().flatten).count(df_min_value)} times;')
    print(
        f'max value in matrix_dataframe {df_max_value} appears {list(df.to_numpy().flatten).count(df_max_value)} times')
    print(
        f'\nmedian value in matrix_dataframe {np.median(matrix_dataframe.values)} mean value in matrix_dataframe {np.mean(matrix_dataframe.values)}')

    list_of_lists_from_df = matrix_dataframe.values.tolist()
    import itertools
    one_big_list_of_values_from_matrix_df = list(itertools.chain.from_iterable(list_of_lists_from_df))
    number_of_different_values = len(set(one_big_list_of_values_from_matrix_df))
    print(
        f'\nnumber of different values in matrix_dataframe is  {number_of_different_values} ')
    
    num_of_values_in_matrix = 33538*4992 #todo: change later if i have a different matrix
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

    print("\n----- finished function load_dataframes_from_mtx_and_tsv -----\n")

    return matrix_dataframe, features_dataframe, barcodes_datafame


def load_dataset_from_images_folder(path_to_images):
    print("\n----- entered function load_dataset_from_pictures_folder -----")

    # fix_image_filenames()  # !!! NOTE: this was executed once to change the file name.

    im_hight_and_width_size = 176  # NOTE <--
    tf = torchTransform.Compose([
        # Resize to constant spatial dimensions
        torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
        # PIL.Image -> torch.Tensor
        torchTransform.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object = ImageFolder(os.path.dirname(path_to_images), tf)

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

    print("\n----- finished function load_dataset_from_pictures_folder -----\n")

    return dataset_object


import torch


class STDL_Dataset(torch.utils.data.Dataset):
    '''

    *** TRY 1 ***

    NOTE: every element of the dataset is a 2d tuple of: (img tensor, gene exp value)

    NOTE: the above gene exp value is for a specific gene

    '''

    def __init__(self, path_to_images_dir, path_to_mtx_tsv_files_dir, chosen_gene_name):
        print("\n----- entering __init__ phase of  STDL_Dataset -----")

        # just in case:
        # path_to_images_dir = "C:/Users/royru/Downloads/spatialGeneExpression/images"  # looks for all sub folders, finds only: # /images/  #
        # path_to_mtx_tsv_files_dir = "C:/Users/royru/Downloads/spatialGeneExpression"

        # self.imageFolder = imageFolder
        self.imageFolder = load_dataset_from_images_folder(path_to_images_dir)
        # self.barcodes_df = barcodes_df
        self.matrix_dataframe, self.features_dataframe, self.barcodes_datafame = load_dataframes_from_mtx_and_tsv_new(
            path_to_mtx_tsv_files_dir)

        # for future usage: #TODO:
        self.gene_name = chosen_gene_name

        print("\n----- finished __init__ phase of  STDL_Dataset -----\n")

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.imageFolder)

    def __getitem__(self, index):
        '''

        # 'Generates one sample of data'
        # Select sample

        Task: attach the y value of a single img

        :param index:
        :return:
        '''

        curr_filename = self.imageFolder.samples[index][0]
        curr_img_tensor = self.imageFolder[index][0]  # note that this calls __get_item__ and returns the tensor value
        # for me
        X = curr_img_tensor  # this is actually X_i

        curr_sample_name = curr_filename.partition('_')[0].partition('\\images\\')[2]  # first partition to get all
        # before the first _ , second partition to get everything after \\images\\

        # get the y value's COLUMN in the gene expression matrix MTX (with help from the barcodes df)
        output_indices_list = self.barcodes_datafame.index[
            self.barcodes_datafame['barcodes'] == curr_sample_name].tolist()
        # assert (len(output_indices_list) == 1) # TODO: check if this is needed
        curr_sample_name_index_in_barcoes_df = output_indices_list[0]
        column = curr_sample_name_index_in_barcoes_df

        # get the y value's ROW in the gene expression matrix MTX (with help from the features df)
        output_indices_list = self.features_dataframe.index[
            self.features_dataframe['gene_names'] == self.gene_name].tolist()
        assert (len(output_indices_list) == 1)
        curr_gene_name_index_in_features_df = output_indices_list[0]
        row = curr_gene_name_index_in_features_df

        # finally, get the y value from the gene expression matrix MTX
        current_gene_expression_value = self.matrix_dataframe.iloc[row, column]

        # for me
        y = current_gene_expression_value

        return X, y
