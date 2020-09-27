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
from sklearn.decomposition import NMF
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import deepNetworkArchitechture
import projectUtilities 

import torch

import cv2
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt


def load_dataframes_from_mtx_and_tsv_new(path_to_mtx_tsv_files_dir):
    '''
    original loading code from website:
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/matrices
    '''

    print("\n----- entered function load_dataframes_from_mtx_and_tsv -----")

    print("started reading features.tsv")
    path_to_features = path_to_mtx_tsv_files_dir + "/features.tsv"  # TODO: note no gz
    features_dataframe = pd.read_csv(path_to_features, sep='\t', header=None)
    features_dataframe.columns = ['feature_ids', 'gene_names', 'feature_types']  # giving columns their names
    print("V  finished reading features.tsv")

    print("started reading barcodes.tsv")
    path_to_barcodes = path_to_mtx_tsv_files_dir + "/barcodes.tsv"  # TODO: note no gz
    barcodes_datafame = pd.read_csv(path_to_barcodes, sep='\t', header=None)
    barcodes_datafame.columns = ['barcodes']  # giving columns their names
    print("V  finished reading barcodes.tsv")
    barcodes = barcodes_datafame['barcodes'].tolist()

    print("started reading matrix.mtx. this might take some time ...")
    path_to_matrix = path_to_mtx_tsv_files_dir + "/matrix.mtx"  # TODO: note no gz
    matrix = scipy.io.mmread(path_to_matrix)
    matrix_dataframe = pd.DataFrame.sparse.from_spmatrix(matrix)  # todo: note sure this works. from: https://pandas.pydata.org/docs/user_guide/sparse.html
    print("V  finished reading matrix.mtx")

    # testing 230920 morning
    print("adjusting matrix_dataframe")
    matrix_dataframe = matrix_dataframe.replace([np.inf, -np.inf], np.nan) # replace all inf values with a NaN value
    matrix_dataframe = matrix_dataframe.fillna(0) #fill all NaN values with 0 ....
    matrix_dataframe = matrix_dataframe.dropna(axis=1, how='all') #drop all columns that have ONLY NaN values
    matrix_dataframe = matrix_dataframe.dropna(axis=0, how='any') #drop all rows that have at least one NaN value
    # matrix_dataframe = matrix_dataframe.astype(int) #convert value types to int
    print("V  finished working on matrix_dataframe")

    # # print information if requested by user
    # yes = {'yes','y', 'ye', '','YES','YE','Y'} # raw_input returns the empty string for "enter"
    # no = {'no','n','NO','N'}
    # # get input from user
    # choice = input("Do you wish to print information about the 3 loaded dataframes ? [yes/no]")  
    # if choice in yes:
    #     projectUtilities.printInfoAboutDFs(matrix_dataframe, features_dataframe, barcodes_datafame)
    # elif choice in no:
    #     pass
    # else:
    #     print("since you did not input a yes, thats a no :)")

    # projectUtilities.printInfoAboutDFs(matrix_dataframe, features_dataframe, barcodes_datafame)
    
    print("\n----- finished function load_dataframes_from_mtx_and_tsv -----\n")

    return matrix_dataframe, features_dataframe, barcodes_datafame


def load_dataset_from_images_folder(path_to_images):
    '''
    NOTE: the dataset refered to here is the imageFolder dataset created from the original images folder
    '''
    print("\n----- entered function load_dataset_from_pictures_folder -----")

    # fix_image_filenames()  # !!! NOTE: this was executed once to change the file name. # TODO

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

    # # print information if requested by user
    # yes = {'yes','y', 'ye', '','YES','YE','Y'} # raw_input returns the empty string for "enter"
    # no = {'no','n','NO','N'}
    # # get input from user
    # choice = input("Do you wish to print information about the ImageFolder dataset object ? [yes/no]")  
    # if choice in yes:
    #     projectUtilities.printInfoAboutImageFolderDataset(dataset_object) 
    # elif choice in no:
    #     pass
    # else:
    #     print("since you did not input a yes, thats a no :)")

    
    print("\n----- finished function load_dataset_from_pictures_folder -----\n")

    return dataset_object


def load_augmented_imageFolder_DS_from_images_folder(path_to_images):
    '''
    NOTE: dont freak out from the large amount of code, most of this function is transformations that are "copy-pasted" with slight differences.
          There are overall 8 differnt transformation - 0/90/180/270 , and the same four flipped horizontaly
    '''
    print("\n----- entered function load_dataset_from_pictures_folder -----")

    # fix_image_filenames()  # !!! NOTE: this was executed once to change the file name. # TODO

    im_hight_and_width_size = 176  # NOTE <--
    
    # note that this next "compose" actually a pipeline
    tf_original =   torchTransform.Compose([
                    # Resize to constant spatial dimensions
                    torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
                    # PIL.Image -> torch.Tensor
                    torchTransform.ToTensor(),
                    # Dynamic range [0,1] -> [-1, 1]
                    torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object_original = ImageFolder(os.path.dirname(path_to_images), tf_original)

    # note that this next "compose" actually a pipeline
    tf_rotated_90 = torchTransform.Compose([
                    # Resize to constant spatial dimensions
                    torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
                    # Rotate image:
                    # NOTE: degrees (sequence or float or int) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
                    torchTransform.RandomRotation((90,90)),
                    # PIL.Image -> torch.Tensor
                    torchTransform.ToTensor(),
                    # Dynamic range [0,1] -> [-1, 1]
                    torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object_90 = ImageFolder(os.path.dirname(path_to_images), tf_rotated_90)

    # note that this next "compose" actually a pipeline
    tf_rotated_180 = torchTransform.Compose([
                    # Resize to constant spatial dimensions
                    torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
                    # Rotate image:
                    # NOTE: degrees (sequence or float or int) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
                    torchTransform.RandomRotation((180,180)),
                    # PIL.Image -> torch.Tensor
                    torchTransform.ToTensor(),
                    # Dynamic range [0,1] -> [-1, 1]
                    torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object_180 = ImageFolder(os.path.dirname(path_to_images), tf_rotated_180)

    # note that this next "compose" actually a pipeline
    tf_rotated_270 = torchTransform.Compose([
                    # Resize to constant spatial dimensions
                    torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
                    # Rotate image:
                    # NOTE: degrees (sequence or float or int) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
                    torchTransform.RandomRotation((270,270)),
                    # PIL.Image -> torch.Tensor
                    torchTransform.ToTensor(),
                    # Dynamic range [0,1] -> [-1, 1]
                    torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object_270 = ImageFolder(os.path.dirname(path_to_images), tf_rotated_270)

        # note that this next "compose" actually a pipeline
    tf_original_flipped =   torchTransform.Compose([
                    # Resize to constant spatial dimensions
                    torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
                    # flip horizontaly (p=1 == probability for flipping is 1 == always flip)
                    torchTransform.RandomHorizontalFlip(p=1),
                    # PIL.Image -> torch.Tensor
                    torchTransform.ToTensor(),
                    # Dynamic range [0,1] -> [-1, 1]
                    torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object_original_flipped = ImageFolder(os.path.dirname(path_to_images), tf_original)

    # note that this next "compose" actually a pipeline
    tf_rotated_90_flipped = torchTransform.Compose([
                    # Resize to constant spatial dimensions
                    torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
                    # flip horizontaly (p=1 == probability for flipping is 1 == always flip)
                    torchTransform.RandomHorizontalFlip(p=1),
                    # Rotate image:
                    # NOTE: degrees (sequence or float or int) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
                    torchTransform.RandomRotation((90,90)),
                    # PIL.Image -> torch.Tensor
                    torchTransform.ToTensor(),
                    # Dynamic range [0,1] -> [-1, 1]
                    torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object_90_flipped = ImageFolder(os.path.dirname(path_to_images), tf_rotated_90)

    # note that this next "compose" actually a pipeline
    tf_rotated_180_flipped = torchTransform.Compose([
                    # Resize to constant spatial dimensions
                    torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
                    # flip horizontaly (p=1 == probability for flipping is 1 == always flip)
                    torchTransform.RandomHorizontalFlip(p=1),
                    # Rotate image:
                    # NOTE: degrees (sequence or float or int) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
                    torchTransform.RandomRotation((180,180)),
                    # PIL.Image -> torch.Tensor
                    torchTransform.ToTensor(),
                    # Dynamic range [0,1] -> [-1, 1]
                    torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object_180_flipped = ImageFolder(os.path.dirname(path_to_images), tf_rotated_180)

    # note that this next "compose" actually a pipeline
    tf_rotated_270_flipped = torchTransform.Compose([
                    # Resize to constant spatial dimensions
                    torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
                    # flip horizontaly (p=1 == probability for flipping is 1 == always flip)
                    torchTransform.RandomHorizontalFlip(p=1),
                    # Rotate image:
                    # NOTE: degrees (sequence or float or int) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
                    torchTransform.RandomRotation((270,270)),
                    # PIL.Image -> torch.Tensor
                    torchTransform.ToTensor(),
                    # Dynamic range [0,1] -> [-1, 1]
                    torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object_270_flipped = ImageFolder(os.path.dirname(path_to_images), tf_rotated_270)

    # now that we finished creating the datasets, we will create a huge new dataset. 
    # important premise - in all roatations, image names remain the same. this is important because this is our mapping to our gene expression values from matrix_dataframe
    datasets_to_concatanate = [dataset_object_original, dataset_object_90, dataset_object_180, dataset_object_270,
                                dataset_object_original_flipped, dataset_object_90_flipped, dataset_object_180_flipped, dataset_object_270_flipped]
    final_dataset_object = STDL_ConcatDataset_of_ImageFolders(datasets_to_concatanate)


    # # print information if requested by user
    # yes = {'yes','y', 'ye', '','YES','YE','Y'} # raw_input returns the empty string for "enter"
    # no = {'no','n','NO','N'}
    # # get input from user
    # choice = input("Do you wish to print information about the ImageFolder dataset object ? [yes/no]")  
    # if choice in yes:
    #     projectUtilities.printInfoAboutCustomConcatanatedImageFolderDataset(final_dataset_object) 
    # elif choice in no:
    #     pass
    # else:
    #     print("since you did not input a yes, thats a no :)")

    
    print("\n----- finished function load_dataset_from_pictures_folder -----\n")

    return final_dataset_object


def cut_empty_genes(orig_df):
    '''
    recieve a matrix df, cut all empty genes, 
    return: 1. the reduced dataframe
            2. indices of all rows that were kept (for research purpose later on)

    assumption: genes (features) are the rows of the df, samples are the columns
    '''
    print(f'cutting all genes (rows) that contain only zeros ...')
    # trick from stack overflow to keep all rows that have at least one nonzero value
    reduced_df = orig_df.loc[(orig_df!=0).any(axis=1)]
    indices_of_kept_rows = list(reduced_df.index.values)
    reduced_df = reduced_df.reset_index()  # this causes a new column to appear
    reduced_df = reduced_df.rename(columns={"index": "original_index_from_matrix_dataframe"})
    mapping = reduced_df[["original_index_from_matrix_dataframe"]]
    reduced_df = reduced_df.drop(columns=["original_index_from_matrix_dataframe"])

    # # print information if requested by user
    # yes = {'yes','y', 'ye', '','YES','YE','Y'} # raw_input returns the empty string for "enter"
    # no = {'no','n','NO','N'}
    # # get input from user
    # choice = input("Do you wish to print information about the reduced dataframe ? [yes/no]")  
    # if choice in yes:
    #     projectUtilities.printInfoAboutReducedDF(reduced_df)
    # elif choice in no:
    #     pass
    # else:
    #     print("since you did not input a yes, thats a no :)")

    # return 
    return reduced_df, mapping 


    '''

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


def cut_genes_with_under_B_counts(orig_df, Base_value):
    '''
    recieve a matrix df, cut all  genes with under Base_value counts in the original matrix_dataframe 
    == keep all genes with over Base_value counts
    return: 1. the reduced dataframe
            2. indices of all rows that were kept (for research purpose later on)

    assumption: genes (features) are the rows of the df, samples are the columns
    '''
    print(f'cutting all genes (rows) that contain less than B counts ...')
    # trick from stack overflow to keep all rows that have at least one nonzero value
    reduced_df = orig_df[orig_df.sum(axis=1) > Base_value]
    indices_of_kept_rows = list(reduced_df.index.values)
    reduced_df = reduced_df.reset_index()  # this causes a new column to appear - "index" which contains the old indices before resetting
    reduced_df = reduced_df.rename(columns={"index": "original_index_from_matrix_dataframe"})
    mapping = reduced_df[["original_index_from_matrix_dataframe"]]
    reduced_df = reduced_df.drop(columns=["original_index_from_matrix_dataframe"])

    # # print information if requested by user
    # yes = {'yes','y', 'ye', '','YES','YE','Y'} # raw_input returns the empty string for "enter"
    # no = {'no','n','NO','N'}
    # # get input from user
    # choice = input("Do you wish to print information about the reduced dataframe ? [yes/no]")  
    # if choice in yes:
    #     projectUtilities.printInfoAboutReducedDF(reduced_df)
    # elif choice in no:
    #     pass
    # else:
    #     print("since you did not input a yes, thats a no :)")

    # projectUtilities.printInfoAboutReducedDF(reduced_df)

    # return 
    return reduced_df, mapping 


def perform_log_1p_normalization(df):
    '''
    perform log 1P normaliztion on the matrix values:
    note that the original dataframe contains "count" values (integers from 0 to max value)
    the transformation of a single value will be as follows:
    (step 1) add +1 to each entry
    (step 2) perform a log transformation for each entry

    according to numpy: 
    > Return the natural logarithm of one plus the input array, element-wise.
    > Calculates log(1 + x).
    '''
    print(f'performing log1P transformation of the dataframe ...\n')
    # step 1 and 2 combined
    df_normalized = df.apply(np.log1p)
    # # print if wanted
    # projectUtilities.printInfoAboutReducedDF(df_normalized)
    return df_normalized


class STDL_Dataset_SingleValuePerImg(torch.utils.data.Dataset):
    '''
    NOTE: every element of the dataset is a 2d tuple of: (img tensor, gene exp value)
    NOTE: the above gene exp value is for a specific gene
    '''

    def __init__(self, imageFolder, matrix_dataframe, features_dataframe, barcodes_datafame, chosen_gene_name, index_mapping):
        print("\n----- entering __init__ phase of  STDL_Dataset_SingleValuePerImg -----")

        # Save important information from outside
        self.imageFolder = imageFolder
        self.matrix_dataframe, self.features_dataframe, self.barcodes_datafame = matrix_dataframe, features_dataframe, barcodes_datafame
        self.gene_name = chosen_gene_name
        self.index_mapping = index_mapping

        # save additional information
        self.num_of_features_matrix_df = len(matrix_dataframe.index) 
        self.num_of_samples_matrix_df = len(matrix_dataframe.columns)
        self.size_of_dataset = len(self.imageFolder) # NOTE: size_of_dataset != num_of_samples  

        '''
        create the reduced dataframe == a dataframe with only one row
        '''
        # get the y value's ROW in the gene expression matrix MTX (with help from the features df)
        output_indices_list = self.features_dataframe.index[self.features_dataframe['gene_names'] == self.gene_name].tolist()
        assert (len(output_indices_list) == 1)
        curr_gene_name_index_in_features_df = output_indices_list[0]
        row_in_original_dataframe = curr_gene_name_index_in_features_df
        
        # this row number needs to be transformed with the mapping because some rows were deleted
        row = self.index_mapping.index[self.index_mapping['original_index_from_matrix_dataframe'] == row_in_original_dataframe].tolist()
        assert (len(row) == 1)  # there should be only one of these
        row = row[0]  # list of size 1
        # verify that the recieved gene name's row number wasnt deleted in `cut_genes_with_under_B_counts` function
        assert len(self.matrix_dataframe.index.values) >= row
        # now we can finaly save the data
        self.row = row
        self.reduced_dataframe = self.matrix_dataframe.iloc[row, :]  # get only the relevant gene's row over ALL samples (== all columns)
        # self.reduced_dataframe.rename( columns={0 :'values'}, inplace=True ) # since the previous line gave us one column of values with no name, I renamed it

        print("\n----- finished __init__ phase of  STDL_Dataset_SingleValuePerImg -----\n")

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.imageFolder)

    def __getitem__(self, index):
        '''

        # 'Generates one sample of data'
        # Select sample

        Task: attach the y value of a single img
        '''
        ## get information about the image depending on the object type
        if hasattr(self.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            curr_filename = self.imageFolder.samples[index][0]
            curr_img_tensor = self.imageFolder[index][0]  # note that this calls __get_item__ and returns the tensor value
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            curr_img_tensor, curr_filename = self.imageFolder[index]

        # for me
        X = curr_img_tensor  # this is actually X_i

        # get the sample's name from its absolute path and file name
        curr_sample_name = curr_filename.partition('_')[0].partition('/images/')[2]  # first partition to get all
        # before the first _ , second partition to get everything after \\images\\
        # NOTE: note that \\images\\  might apply for windows addresses only... !?
        #                 /images/  is in LINUX !!!!!!

        # get the y value's COLUMN in the gene expression matrix MTX (with help from the barcodes df)
        output_indices_list = self.barcodes_datafame.index[self.barcodes_datafame['barcodes'] == curr_sample_name].tolist()
        curr_sample_name_index_in_barcoes_df = output_indices_list[0]
        column = curr_sample_name_index_in_barcoes_df

        # get the y value's ROW in the gene expression matrix MTX 
        current_gene_expression_value = self.matrix_dataframe.iloc[self.row, column]

        # TODO: the block below might not be needed
        # NOTE: there should be only one row in the reduced dataframe -
        #       and when this happens then the dataframe is transpoed and columns become rows
        #       and so to get the index at column C its as if we need to get info from row C
        #       and since now after transpose there is only 1 column, then the iloc on the entire row returns only 1 value
        # current_gene_expression_value = self.reduced_dataframe.iloc[[column]]

        # for me
        y = current_gene_expression_value

        return X, y, column  # note that "column" is here for future reference, and is the column in matrix_dataframe that this y value belongs to


class STDL_Dataset_KValuesPerImg_KGenesWithHighestVariance(torch.utils.data.Dataset):
    '''
    NOTE: every element of the dataset is a 2d tuple of: (img tensor, k-dim tensor)  ** the tensor is from k-dim latent space
    '''

    def __init__(self, imageFolder, matrix_dataframe, features_dataframe, barcodes_datafame, num_of_dims_k):
        print("\n----- entering __init__ phase of  STDL_Dataset_KValuesPerImg_KGenesWithHighestVariance -----")

        # Save important information from outside
        self.imageFolder = imageFolder
        self.matrix_dataframe, self.features_dataframe, self.barcodes_datafame = matrix_dataframe, features_dataframe, barcodes_datafame
        # NOTE: the matrix_dataframe above is a reduced version of the original matrix_dataframe
        self.num_of_dims_k = num_of_dims_k

        # save additional information
        self.num_of_features_matrix_df = len(matrix_dataframe.index) 
        self.num_of_samples_matrix_df = len(matrix_dataframe.columns)
        self.size_of_dataset = len(self.imageFolder) # NOTE: size_of_dataset != num_of_samples  

        '''
        create a reduced dataframe == a dataframe with only K chosen features 
        '''
        print("calculate variance of all columns from  matrix_dataframe - and choosing K genes with higest variance ...")
        import pandas as pd
        variance_df = matrix_dataframe.var(axis=1)  # get the variance of all the genes [varience of each gene over all samples] 
        # (this practically 'flattens' the df to one column of 33538 entries - one entry for each gene over all the samples)
        variance_df = pd.DataFrame(data=variance_df, columns=['variance']) # convert it from a pandas series to a pandas df

        # df.nlargest - This method is equivalent to df.sort_values(columns, ascending=False).head(n), but more performant.
        nlargest_variance_df = variance_df.nlargest(n=num_of_dims_k, columns=['variance'], keep='all')
        # now use the indexes recieved above to retrieve the entries with the highest variance
        list_of_nlargest_indices = list(nlargest_variance_df.index.values) 
        # save it for future reference
        self.list_of_nlargest_indices = list_of_nlargest_indices
        # #save 
        # TODO: previously (changed 220920 evening)
        #self.reduced_dataframe = matrix_dataframe.iloc[list_of_nlargest_indices , :]  # get k rows (genes) with highest variance over all of the columns         
        # new:
        reduced_df = matrix_dataframe.iloc[list_of_nlargest_indices , :]  # get k rows (genes) with highest variance over all of the columns  
        reduced_df = reduced_df.reset_index()  # this causes a new column to appear - "index" which contains the old indices before resetting
        reduced_df = reduced_df.rename(columns={"index": "original_index_from_matrix_dataframe"})
        self.mapping = reduced_df[["original_index_from_matrix_dataframe"]]
        reduced_df = reduced_df.drop(columns=["original_index_from_matrix_dataframe"])
        self.reduced_dataframe = reduced_df

        print("\n----- finished __init__ phase of  STDL_Dataset_LatentTensor -----\n")

    def __len__(self):
        # 'Denotes the total number of samples (that actually have images attached to them'
        return len(self.imageFolder)

    def __getitem__(self, index):
        '''

        # 'Generates one sample of data'
        # Select sample

        Task: attach the y value of a single img
            NOTE: the y value is now a k-dim tensor
                    this practiclay means that we extract a column from the df as our y value

        :param index:
        :return:
        '''

        if hasattr(self.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            curr_filename = self.imageFolder.samples[index][0]
            curr_img_tensor = self.imageFolder[index][0]  # note that this calls __get_item__ and returns the tensor value
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            curr_img_tensor, curr_filename = self.imageFolder[index]

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

        # note that unlike before, we want to get the y value of the ENTIRE COLUMN of the REDUCED dataframe (reduced_df should be  K x 4992)
        # and so, return the entire column
        current_gene_expression_values = self.reduced_dataframe.iloc[:, column] 

        # for me
        y = current_gene_expression_values
        # convert to a tensor
        y = torch.from_numpy(y.to_numpy()).float()  # assumption: currently "y" is a pandas series (\dataframe) slice, and needs to be converted to a tensor for later usage
        assert (len(y) == self.num_of_dims_k)
        # TODO: check: currently y is a numpy ndarray type. does it need to be switched to a tensor type ??? <-----------------------------------

        return X, y, column  # note that "column" is here for future reference, and is the column in matrix_dataframe that these y values belong to


class STDL_Dataset_KValuesPerImg_LatentTensor_NMF(torch.utils.data.Dataset):
    '''
    NOTE: every element of the dataset is a 2d tuple of: (img tensor, k-dim tensor)  ** the tensor is from k-dim latent space
    '''

    def __init__(self, imageFolder, matrix_dataframe, features_dataframe, barcodes_datafame, num_of_dims_k):
        print("\n----- entering __init__ phase of  STDL_Dataset_KValuesPerImg_LatentTensor_NMF -----")

        # Save important information from outside
        self.imageFolder = imageFolder
        self.matrix_dataframe, self.features_dataframe, self.barcodes_datafame = matrix_dataframe, features_dataframe, barcodes_datafame
        # NOTE: the matrix_dataframe above is a reduced version of the original matrix_dataframe after cutting features by condition
        self.num_of_dims_k = num_of_dims_k

        # save additional information
        self.num_of_features_matrix_df = len(matrix_dataframe.index) 
        self.num_of_samples_matrix_df = len(matrix_dataframe.columns)
        self.size_of_dataset = len(self.imageFolder) # NOTE: size_of_dataset != num_of_samples  

        # create the reduced dataframe:
        print("performing NMF decomposition on main matrix dataframe ...")
        self.nmf_model = NMF(n_components=num_of_dims_k, init='random', random_state=0)  # TODO: what init should we use here ?
        self.W = self.nmf_model.fit_transform(matrix_dataframe)  # TODO: check if we need a "fit transform" here or not
        self.H = self.nmf_model.components_
        # !!! create the reduced dataframe !!!
        self.reduced_dataframe = self.H   
        self.reduced_dataframe = pd.DataFrame(data=self.reduced_dataframe)



        print("\n----- finished __init__ phase of  STDL_Dataset_LatentTensor -----\n")


    def __len__(self):
        # 'Denotes the total number of samples (that actually have images attached to them'
        return self.size_of_dataset


    def __getitem__(self, index):
        '''

        # 'Generates one sample of data'
        # Select sample

        Task: attach the y value of a single img
            NOTE: the y value is now a k-dim tensor
                    this practiclay means that we get a row

        :param index:
        :return:
        '''

        if hasattr(self.imageFolder, 'samples'):  # if the image folder has a field called "samples", it means that this is a regular "ImageFolder" type
            curr_filename = self.imageFolder.samples[index][0]
            curr_img_tensor = self.imageFolder[index][0]  # note that this calls __get_item__ and returns the tensor value
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            curr_img_tensor, curr_filename = self.imageFolder[index]

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

        # note that unlike before, we want to get the y value of the ENTIRE COLUMN of the REDUCED dataframe (reduced_df should be  K x 4992)
        # and so, return the entire column

        current_k_dim_vector = self.reduced_dataframe.iloc[:, column] 

        # for me
        y = current_k_dim_vector
        # convert to a tensor
        y = torch.from_numpy(y.to_numpy()).float()  # assumption: currently "y" is a pandas series (\dataframe) slice, and needs to be converted to a tensor for later usage
        assert (len(y) == self.num_of_dims_k)
        # TODO: check: currently y is a numpy ndarray type. does it need to be switched to a tensor type ??? <-----------------------------------

        return X, y, column  # note that "column" is here for future reference, and is the column in matrix_dataframe that these y values belong to


class STDL_Dataset_KValuesPerImg_LatentTensor_AutoEncoder(torch.utils.data.Dataset):
    '''
    NOTE: every element of the dataset is a 2d tuple of: (img tensor, k-dim tensor)  ** the tensor is from k-dim latent space
    '''

    def __init__(self, imageFolder, matrix_dataframe, features_dataframe, barcodes_datafame, num_of_dims_k, device):
        print("\n----- entering __init__ phase of  STDL_Dataset_KValuesPerImg_LatentTensor_AutoEncoder -----")

        # Save important information from outside
        self.imageFolder = imageFolder
        self.matrix_dataframe, self.features_dataframe, self.barcodes_datafame = matrix_dataframe, features_dataframe, barcodes_datafame
        # NOTE: the matrix_dataframe above is a reduced version of the original matrix_dataframe
        self.num_of_dims_k = num_of_dims_k

        # save additional information
        self.num_of_features_matrix_df = len(matrix_dataframe.index) 
        self.num_of_samples_matrix_df = len(matrix_dataframe.columns)
        self.size_of_dataset = len(self.imageFolder) # NOTE: size_of_dataset != num_of_samples  

        # initialize the autoencoder
        print("initializing the autoencoder (this might take a while) ...")
        num_of_rows_in_matrix_df = len(matrix_dataframe.index)
        num_of_features = num_of_rows_in_matrix_df
        num_of_columns_in_matrix_df = len(matrix_dataframe.columns)
        
        self.autoEncoder = self.return_trained_AE_net(in_features=num_of_features, z_dim=num_of_dims_k, device=device)

        print("\n----- finished __init__ phase of  STDL_Dataset_KValuesPerImg_LatentTensor_AutoEncoder -----\n")


    def __len__(self):
        # 'Denotes the total number of samples (that actually have images attached to them)' - see __init__
        return self.size_of_dataset


    def __getitem__(self, index):
        '''

        # 'Generates one sample of data'
        # Select sample

        Task: attach the y value of a single img
            NOTE: the y value is now a k-dim tensor
                    this practiclay means that we get a row

        :param index:
        :return:
        '''

        if hasattr(self.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            curr_filename = self.imageFolder.samples[index][0]
            curr_img_tensor = self.imageFolder[index][0]  # note that this calls __get_item__ and returns the tensor value
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            curr_img_tensor, curr_filename = self.imageFolder[index]

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

        # note that unlike before, we want to get the y value of the ENTIRE COLUMN of the REDUCED dataframe (reduced_df should be  K x 4992)
        # and so, return the entire column
        vector_with_all_features = self.matrix_dataframe.iloc[:, column].to_numpy()  
        # convert it to torch
        vector_with_all_features = torch.from_numpy(vector_with_all_features).float().cuda()  #NOTE: note the cuda here !!!
        # encode the vector == convert to a z dimensional latent vector
        latent_vector = self.autoEncoder.encodeWrapper(vector_with_all_features)            #TODO: maybe the vector is numpy and needs to be converted to a tensor

        # for me
        y = latent_vector.squeeze()  # note the squeeze ! this is relevant because the batch size  of "latent_vector" is 1
        assert (len(y) == self.num_of_dims_k)
        # TODO: check: currently y is a numpy ndarray type. does it need to be switched to a tensor type ??? <-----------------------------------

        return X, y, column  # note that "column" is here for future reference, and is the column in matrix_dataframe that these y values belong to


    def return_trained_AE_net(self, in_features, z_dim, device):
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
        dataset = STDL_Dataset_matrix_df_for_AE_init(self.matrix_dataframe)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
        x0 = dataset[0]
        num_of_features = len(x0)

        '''
        prepare model, loss and optimizer instances
        '''

        # model
        connected_layers_dim_list = [100*z_dim, 10*z_dim, 5*z_dim]  #NOTE: this is without the first and last layers !
        print(f'note - number of (hidden) linear layers is supposed to be {len(connected_layers_dim_list)}')
        model = deepNetworkArchitechture.AutoencoderNet(in_features=num_of_features, connected_layers_dim_list=connected_layers_dim_list, z_dim=z_dim, batch_size=batch_size, device=device)
        # TODO: this next if condition migt be temporarily commented because i get CUDA OUT OF MEMORY errors. (it shouldnt be commented)
        if device.type == 'cuda':
            model = model.to(device=device)  # 030920 test: added cuda
        
        # loss and optimizer
        loss_fn = torch.nn.MSELoss()
        learning_rate = 1e-4  #TODO: need to play with this value ....
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        '''
        now we can perform the training
        '''

        print("****** begin training ******")
        num_of_epochs = 3  # TODO: this number was lowered since i dont have alot of time to play around :)
        max_alowed_number_of_batches = 999999  # the purpose of this var is if i dont realy want all of the batches to be trained uppon ... 
        num_of_batches = (len(dataset) // batch_size)  # TODO: check this line
        if num_of_batches > max_alowed_number_of_batches:
            print(f'NOTE: in order to speed up training (while damaging accuracy) the number of batches per epoch was reduced from {num_of_batches} to {max_alowed_number_of_batches}')
            num_of_batches = max_alowed_number_of_batches


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
                loss = loss_fn(x_reconstructed, x)  # TODO: check order and correctness
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

            # # verification prints:
            # print(f'loss_values_list len {len(loss_values_list)}')
            # print(f'loss_values_list is {loss_values_list}')

            # data prints on the epoch that ended
            print(f'in this epoch: min loss {np.min(loss_values_list)} max loss {np.max(loss_values_list)}')
            print(f'               average loss {np.mean(loss_values_list)}')

        pass
        print("\n----- finished function return_trained_AE_net -----\n")

        # return the trained model TODO: check this is the correct syntax
        return model


class STDL_Dataset_matrix_df_for_AE_init(torch.utils.data.Dataset):
    '''
    This dataset holds information about the matrix dataframe, and will be used for the initialization
    of the autoencoder network inside of "STDL_Dataset_KValuesPerImg_LatentTensor_AutoEncoder".
    '''
    def __init__(self, matrix_df):
        self.data = matrix_df.to_numpy()
        self.num_of_samples = len(matrix_df.columns)
        self.num_of_features = len(matrix_df.index)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, index):
        x = self.data[:, index]  # note - this is written this way since we want a COLUMN from the matrix_df. (note: should by shape  (num_features, 1) )
                                                                                                                                    #  num_genes     
        return x


class STDL_ConcatDataset_of_ImageFolders(torch.utils.data.Dataset):
    '''
    This is a concatanation of ImageFolder datasets into one unified dataset. 
    NOTE: the assumption is that the list of datastes recieved as input for the __init__ method are all "ImageFolder", and all have the same size
                but different transformations
    '''
    def __init__(self, datasets_list):
        self.datasets_list = datasets_list
        self.dataset_lengths_list = [len(ds) for ds in datasets_list]
        self.index_offsets = np.cumsum(self.dataset_lengths_list)  # cumsum = cumulative sum. this returns a list (length of datasets_list) in which every element is a cumulative sum of the length that far.
                                                                                              # say the original DS is size 30. then this returns: [30,60,90,120,...]
        self.total_size = np.sum(self.dataset_lengths_list)
        
        # because all of the datasets are supposed to be the same images but transformed differently:
        self.single_dataset_length = self.dataset_lengths_list[0]  #note that all datasets are supposed to be the same length
        self.list_of_image_filenames = [filename for (filename, not_relevant) in self.datasets_list[0].samples]  #note that all datasets are supposed to be the same length


    def __len__(self):
        return self.total_size


    def __getitem__(self, index):
        '''
        note:  index (param) is for in the range of the entire concatanated DS
        '''
        final_index_in_ds = index
        for dataset_index, offset in enumerate(self.index_offsets):
            if index < offset:
                # if needed (if > 0) adjust index inside the wanted ds according to the cummulative index offsets
                if dataset_index > 0:  
                    final_index_in_ds = index - self.index_offsets[dataset_index-1]
                # prepare information to return
                curr_filename = self.list_of_image_filenames[final_index_in_ds]
                curr_img_tensor = self.datasets_list[dataset_index][final_index_in_ds][0]
                return curr_img_tensor, curr_filename
            else: 
                pass
        # if we got here, the index is invalid
        raise IndexError(f'{index} exceeds {self.length}')


def create_smaller_images_from_biopsy_sample(path_to_dir):
    '''
    Function to create_smaller_images_from_biopsy_sample
    '''
    print("\n----- entered function create_smaller_images_from_biopsy_sample -----")

    # data points(x,y coordinate) for the tissue
    path1 = path_to_dir + "/tissue_positions_list.csv"
    positions_dataframe = pd.read_csv(path1,names=['barcode','tissue','row','col','x','y'])
    print(f'path1:\n {path1}')

    # import image: comes with BGR
    path2 = path_to_dir + "/V1_Breast_Cancer_Block_A_Section_2_image.tif"
    print(f'path2:\n {path2}')
    img = cv2.imread(path2)
    print(f'img.type {type(img)} ')
    print(f'img.shape {img.shape} ')

    # output path
    out_path = path_to_dir + "/images/"

    # crate the output folder if it doesnt exists
    import os
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ## TODO - get from "scalefactors_json" file.
    # diameter & diameter for image 
    spot_diameter_fullres = 177.4829519178534
    spot_radius = int(spot_diameter_fullres/2)

    # num of iterations
    total_amount_of_spots = len(positions_dataframe.index)

    for idx, row in positions_dataframe.iterrows():
        if not row['tissue']:
            continue
        barcode = row['barcode']
        x = row['x']
        y = row['y']       

        #file names
        square_file_name = "{}_x{}_y{}_square.png".format(barcode,x,y)  # previously: '{}_x{}_y{}_{}_square.png'.format(idx,x,y,barcode)

        #print progress
        print(f'processing image {idx} of {total_amount_of_spots} with name: {square_file_name}', end='\r')

        # square image
        roi_square = img[y-spot_radius:y+spot_radius, x-spot_radius:x+spot_radius]
        # plt.imshow(roi_square)
        cv2.imwrite(out_path + square_file_name, roi_square)

    pass
    print(f'\nfinished cutting the big image')
    

