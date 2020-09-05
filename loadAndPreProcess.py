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

import projectUtilities 

def load_dataframes_from_mtx_and_tsv_new(path_to_mtx_tsv_files_dir):
    '''
    original loading code from website:
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/matrices
    '''

    print("\n----- entered function load_dataframes_from_mtx_and_tsv -----")

    path_to_matrix = path_to_mtx_tsv_files_dir + "/matrix.mtx"  # TODO: note no gz
    matrix = scipy.io.mmread(path_to_matrix)
    matrix_dataframe = pd.DataFrame.sparse.from_spmatrix(
        matrix)  # todo: note sure this works. from: https://pandas.pydata.org/docs/user_guide/sparse.html
    print("-DBG-: finished reading matrix.mtx")

    path_to_features = path_to_mtx_tsv_files_dir + "/features.tsv"  # TODO: note no gz
    features_dataframe = pd.read_csv(path_to_features, sep='\t', header=None)
    features_dataframe.columns = ['feature_ids', 'gene_names', 'feature_types']  # giving columns their names
    print("-DBG-: finished reading features.tsv")

    path_to_barcodes = path_to_mtx_tsv_files_dir + "/barcodes.tsv"  # TODO: note no gz
    barcodes_datafame = pd.read_csv(path_to_barcodes, sep='\t', header=None)
    barcodes_datafame.columns = ['barcodes']  # giving columns their names
    print("-DBG-: finished reading barcodes.tsv")
    barcodes = barcodes_datafame['barcodes'].tolist()

    # print information if requested by user
    yes = {'yes','y', 'ye', '','YES','YE','Y'} # raw_input returns the empty string for "enter"
    no = {'no','n','NO','N'}
    # get input from user
    choice = input("Do you wish to print information about the 3 loaded dataframes ? [yes/no]")  
    if choice in yes:
        projectUtilities.printInfoAboutDFs(matrix_dataframe, features_dataframe, barcodes_datafame)
    elif choice in no:
        pass
    else:
        print("since you did not input a yes, thats a no :)")

    
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

    # print information if requested by user
    yes = {'yes','y', 'ye', '','YES','YE','Y'} # raw_input returns the empty string for "enter"
    no = {'no','n','NO','N'}
    # get input from user
    choice = input("Do you wish to print information about the ImageFolder dataset object ? [yes/no]")  
    if choice in yes:
        #projectUtilities.printInfoAboutCustomDataset(dataset_object)  #TODO: this is not working for some strange reason ...
        print("currently not working... will be resolved in the near future")
    elif choice in no:
        pass
    else:
        print("since you did not input a yes, thats a no :)")

    
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


        curr_sample_name = curr_filename.partition('_')[0].partition('/images/')[2]  # first partition to get all
        # before the first _ , second partition to get everything after \\images\\
        # TODO: note that \\images\\  might apply for windows addresses only... !?
        #                 /images/  is in LINUX !!!!!!

        # get the y value's COLUMN in the gene expression matrix MTX (with help from the barcodes df)
        output_indices_list = self.barcodes_datafame.index[self.barcodes_datafame['barcodes'] == curr_sample_name].tolist()
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
