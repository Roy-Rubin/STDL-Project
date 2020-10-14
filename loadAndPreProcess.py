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
    path_to_features = path_to_mtx_tsv_files_dir + "/features.tsv"  
    features_dataframe = pd.read_csv(path_to_features, sep='\t', header=None)
    features_dataframe.columns = ['feature_ids', 'gene_names', 'feature_types']  # giving columns their names
    print("V  finished reading features.tsv")

    print("started reading barcodes.tsv")
    path_to_barcodes = path_to_mtx_tsv_files_dir + "/barcodes.tsv"  
    barcodes_dataframe = pd.read_csv(path_to_barcodes, sep='\t', header=None)
    barcodes_dataframe.columns = ['barcodes']  # giving columns their names
    print("V  finished reading barcodes.tsv")
    barcodes = barcodes_dataframe['barcodes'].tolist()

    print("started reading matrix.mtx. this might take some time ...")
    path_to_matrix = path_to_mtx_tsv_files_dir + "/matrix.mtx"  
    matrix = scipy.io.mmread(path_to_matrix)
    matrix_dataframe = pd.DataFrame.sparse.from_spmatrix(matrix)  # NOTE: from: https://pandas.pydata.org/docs/user_guide/sparse.html
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
    #     projectUtilities.printInfoAboutDFs(matrix_dataframe, features_dataframe, barcodes_dataframe)
    # elif choice in no:
    #     pass
    # else:
    #     print("since you did not input a yes, thats a no :)")

    # projectUtilities.printInfoAboutDFs(matrix_dataframe, features_dataframe, barcodes_dataframe)
    
    print("\n----- finished function load_dataframes_from_mtx_and_tsv -----\n")

    return matrix_dataframe, features_dataframe, barcodes_dataframe


def load_dataset_from_images_folder(path_to_images, im_hight_and_width_size):
    '''
    NOTE: the dataset refered to here is the imageFolder dataset created from the original images folder
    '''
    print("\n----- entered function load_dataset_from_pictures_folder -----")

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


def load_augmented_imageFolder_DS_from_images_folder(path_to_images, im_hight_and_width_size):
    '''
    NOTE: dont freak out from the large amount of code, most of this function is transformations that are "copy-pasted" with slight differences.
          There are overall 8 differnt transformation - 0/90/180/270 , and the same four flipped horizontaly
    '''
    print("\n----- entered function load_dataset_from_pictures_folder -----")
    
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


def cut_genes_with_under_B_counts_from_train_and_test(train_df, test_df, Base_value):
    '''
    recieve 2 matrix df, cut all genes with under Base_value counts in the original matrix_dataframe 
    == keep all genes with over Base_value counts.

    NOTE: both dataframes need to be with same genes for later purposes.
            and so, we will check who are the genes with under B counts from BOTH dataframes - 
            and remove ONLY the ones that are with under B counts in BOTH of them.

    return: 1. the reduced dataframes
            2. indices of all rows that were kept (for testing purpose later on)

    assumption: genes (features) are the rows of the df, samples are the columns

    TODO: VERIFY: assumption: all genes are in the same order in both dataframes !!!! (check in features_df_train & features_df_test)

    '''
    print(f'checking for genes (rows) that contain less than B counts in both dataframes ...')
    # trick from stack overflow to keep all rows that in them the sum of th values is bigger than the base value given
    temp_df_train = train_df[train_df.sum(axis=1) > Base_value]
    temp_df_test = test_df[test_df.sum(axis=1) > Base_value]
    # get lists of indices
    indices_of_kept_rows_train = list(temp_df_train.index.values)
    indices_of_kept_rows_test = list(temp_df_test.index.values)
    # get the intersection of the lists
    rows_in_intersection = [value for value in indices_of_kept_rows_train if value in indices_of_kept_rows_test]
    rows_in_intersection.sort() # sort() changes the list directly and doesn't return any value
    
    print(f'discarding relevant rows ...')
    ## create the updated dataframes and the mapping (same for both dataframes since we deleted the same rows):
    # keep only wanted rows
    reduced_df_train = train_df.iloc[rows_in_intersection, :]  # reduced_df_train = train_df[rows_in_intersection]
    reduced_df_test = test_df.iloc[rows_in_intersection, :]  # reduced_df_test = test_df[rows_in_intersection]
    # reset indices and create index column
    reduced_df_train = reduced_df_train.reset_index()  # this causes a new column to appear - "index" which contains the old indices before resetting
    reduced_df_test = reduced_df_test.reset_index()  # this causes a new column to appear - "index" which contains the old indices before resetting
    # rename columns
    reduced_df_train = reduced_df_train.rename(columns={"index": "original_index_from_matrix_dataframe"})
    reduced_df_test = reduced_df_test.rename(columns={"index": "original_index_from_matrix_dataframe"})
    # create mapping
    mapping = reduced_df_train[["original_index_from_matrix_dataframe"]]  # this is exactly the same for train and test, BAHAC used train
    # drop the old index column
    reduced_df_train = reduced_df_train.drop(columns=["original_index_from_matrix_dataframe"])
    reduced_df_test = reduced_df_test.drop(columns=["original_index_from_matrix_dataframe"])
    
    # return 
    return reduced_df_train, reduced_df_test, mapping 


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


def cut_samples_with_no_matching_image_and_reorder_df(matrix_df, image_folder_of_the_df, barcodes_df):
    '''
    cut the samples (columns) from the given matrix dataframe that do not have matching images in the given image folder
    NOTE: important assumption: this function should not recieve the augmented image folder, only the regular one

    the other task performed here is reordering the dataframes columns by the indices of the images in the image folder !
    this will help us compare predictions later on in our trainning and testing phase

    example: 
    order of columns in matrix_df:   
                                        0  1  2  3  4  5    
    lets say that out of them 1 and 4 dont exists in the image folder
    we go over the image folder **BY ORDER** in the for loop, and get a column list [2,0,3,5]
    performing `updated_df = matrix_df.iloc[:,column_list]` will give us the dataframe with the COLUMNS ORDERED as:
                                        2  0  3  5
    then the indices will be reset and a mapping will be created
                            old:   2  0  3  5
                            new:   0  1  2  3
    the important thing is that they are **ORDERED** just like the imageFolder order since that is how we created `column_list` !
    '''
    print(f'cutting samples that dont have mathching images in the image folder from the dataframe ...')

    # verify that this is a regular (and not augmented) image folder
    if not hasattr(image_folder_of_the_df, 'samples'):  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
        raise NameError(' wrong image folder type... insert the regular, not augmented one ')
    
    #
    list_of_index_tuples = [] # each element in the list will be a tuple containing (index in image folder, column index in the orig df, column index in the new df)    

    # get indices of samples that DO exist in the image folder, add them to `column_list`
    column_list = []
    for index_in_image_folder, element in enumerate(image_folder_of_the_df.samples):
        filename = element[0]
        curr_sample_name = filename.partition('_')[0].partition('/images/')[2]  # first partition to get everything before the first _ , second partition to get everything after /images/
        index_in_barcoes_df = barcodes_df.index[barcodes_df['barcodes'] == curr_sample_name].item() # assumption - returns only 1 item
        column_list.append(index_in_barcoes_df)
        # in addition, keep track of the connection between indices  
        list_of_index_tuples.append([index_in_image_folder, index_in_barcoes_df, -1]) # -1 will be filled later with the mapping

    # save a mapping between the indices, and save the updated dataframe
    updated_df = matrix_df.iloc[:,column_list] # keep only the wanted columns from the dataframe # NOTE !!!!! this line returns a dataframe in which the columns are ORDERED BY THE ORDER OF `column_list` !!!!
    mapping = updated_df.T.reset_index(drop=False) # drop=False creates an index column. # note the T (transpose) made here. this is because reset_index only works on rows.
    mapping = mapping.rename(columns={"index": "original_index_from_matrix_dataframe"})  
    mapping = mapping['original_index_from_matrix_dataframe']
    mapping = pd.DataFrame(data=mapping, columns=['original_index_from_matrix_dataframe']) # convert it from a pandas series to a pandas df

    # reset columns' indices in the matrix dataframe to be from the ordered range 0 -> n (practically, we are renaming them)
    temp_len = len(updated_df.columns)
    temp_columns_index_list = list(range(0,temp_len))
    updated_df.columns = temp_columns_index_list

    # return both updated_df and mapping
    column_mapping = mapping  # for me
    print(f'V   done :)\n')
    return updated_df, column_mapping


class STDL_Dataset_SingleValuePerImg(torch.utils.data.Dataset):
    '''
    NOTE: every element of the dataset is a 2d tuple of: (img tensor, gene exp value)
    NOTE: the above gene exp value is for a specific gene
    '''

    def __init__(self, imageFolder, matrix_dataframe, features_dataframe, barcodes_dataframe, chosen_gene_name, row_mapping, column_mapping):
        print("\n----- entering __init__ phase of  STDL_Dataset_SingleValuePerImg -----")

        # Save important information from outside
        self.imageFolder = imageFolder
        self.matrix_dataframe, self.features_dataframe, self.barcodes_dataframe = matrix_dataframe, features_dataframe, barcodes_dataframe
        self.gene_name = chosen_gene_name
        self.row_mapping, self.column_mapping = row_mapping, column_mapping

        # save additional information
        self.num_of_features_matrix_df = len(matrix_dataframe.index) 
        self.num_of_samples_matrix_df = len(matrix_dataframe.columns)
        self.size_of_dataset = len(self.imageFolder) # NOTE: size_of_dataset != num_of_samples  
        # 290920 save for later use
        if hasattr(self.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            self.num_of_images_with_no_augmentation = self.size_of_dataset
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            self.num_of_images_with_no_augmentation = imageFolder.dataset_lengths_list[0] # NOTE: the concatanated dataset has the original list of datasets inside it. first in that list is the original untransformed imageFolder DS

        '''
        create the reduced dataframe == a dataframe with only one row
        '''
        # get the y value's ROW in the gene expression matrix MTX (with help from the features df)
        row_in_original_dataframe = self.features_dataframe.index[self.features_dataframe['gene_names'] == self.gene_name].item() # assumption: only one item is returned
    
        # this row number needs to be transformed with the mapping because some rows were deleted
        row = self.row_mapping.index[self.row_mapping['original_index_from_matrix_dataframe'] == row_in_original_dataframe].item() # assumption: only one item is returned


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
        curr_sample_name = curr_filename.partition('_')[0].partition('/images/')[2]  # first partition to get everything before the first _ , second partition to get everything after /images/

        # get the y value's COLUMN in the gene expression matrix df (with help from the barcodes df)
        index_in_barcoes_df = self.barcodes_dataframe.index[self.barcodes_dataframe['barcodes'] == curr_sample_name].item() # assumption: only 1 item is returned
        column = self.column_mapping.index[self.column_mapping['original_index_from_matrix_dataframe'] == index_in_barcoes_df].item() # assumption: only one item is returned

        # get the y value's ROW in the gene expression matrix MTX 
        current_gene_expression_value = self.matrix_dataframe.iloc[self.row, column]

        # for me
        y = current_gene_expression_value

        return X, y, column  # note that "column" is here for future reference, and is the column in matrix_dataframe that this y value belongs to


class STDL_Dataset_KValuesPerImg_KGenesWithHighestVariance(torch.utils.data.Dataset):
    '''
    NOTE: every element of the dataset is a 2d tuple of: (img tensor, k-dim tensor)  ** the tensor is from k-dim latent space
    '''

    def __init__(self, imageFolder, matrix_dataframe, features_dataframe, barcodes_dataframe, column_mapping, row_mapping, num_of_dims_k, k_row_indices=None):
        '''
        NOTE: `k_row_indices=None` added on 141020. its purpose is to create a TESTING reduced dataset that has the same rows in the same order 
                                                     as in the TRAIN reduced dataset after the K genes with highest variance were chosen from it.
                                                     if `None` is passed it means this is the training DS.
                                                     if a list is passed, it means this is the testing DS.
        '''
        print("\n----- entering __init__ phase of  STDL_Dataset_KValuesPerImg_KGenesWithHighestVariance -----")

        # Save important information from outside
        self.imageFolder = imageFolder
        self.matrix_dataframe, self.features_dataframe, self.barcodes_dataframe = matrix_dataframe, features_dataframe, barcodes_dataframe
        self.column_mapping = column_mapping
        self.row_mapping = row_mapping
        # NOTE: the matrix_dataframe above is a reduced version of the original matrix_dataframe
        self.num_of_dims_k = num_of_dims_k

        # save additional information
        self.num_of_features_matrix_df = len(matrix_dataframe.index) 
        self.num_of_samples_matrix_df = len(matrix_dataframe.columns)
        self.size_of_dataset = len(self.imageFolder) # NOTE: size_of_dataset != num_of_samples  
        if hasattr(self.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            self.num_of_images_with_no_augmentation = self.size_of_dataset
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            self.num_of_images_with_no_augmentation = imageFolder.dataset_lengths_list[0] # NOTE: the concatanated dataset has the original list of datasets inside it. first in that list is the original untransformed imageFolder DS

        '''
        create a reduced dataframe == a dataframe with only K chosen features 
        '''        
        # TODO: addition  141020
        if k_row_indices is None:  # meaning we are currently creating the TRAIN DS

            print("calculate variance of all columns from  matrix_dataframe - and choosing K genes with higest variance ...")
            variance_df = matrix_dataframe.var(axis=1)  # get the variance of all the genes [varience of each gene over all samples] 
            # (this practically 'flattens' the df to one column of 33538 entries - one entry for each gene over all the samples)
            variance_df = pd.DataFrame(data=variance_df, columns=['variance']) # convert it from a pandas series to a pandas df

            print(f'--delete-- variance_df {variance_df}')

            # df.nlargest - This method is equivalent to df.sort_values(columns, ascending=False).head(n), but more performant.
            nlargest_variance_df = variance_df.nlargest(n=num_of_dims_k, columns=['variance'], keep='all')

            print(f'--delete-- nlargest_variance_df {nlargest_variance_df}')

            # now use the indexes recieved above to retrieve the entries with the highest variance
            list_of_nlargest_indices = list(nlargest_variance_df.index.values) 
            # save it for future reference
            self.list_of_nlargest_indices = list_of_nlargest_indices

            print(f'--delete-- list_of_nlargest_indices {list_of_nlargest_indices}')

            # #save  
            reduced_df = matrix_dataframe.iloc[list_of_nlargest_indices , :]  # get k rows (genes) with highest variance over all of the columns  
            reduced_df = reduced_df.reset_index()  # this causes a new column to appear - "index" which contains the old indices before resetting
            reduced_df = reduced_df.rename(columns={"index": "original_index_from_matrix_dataframe"})
            self.mapping = reduced_df[["original_index_from_matrix_dataframe"]]
            reduced_df = reduced_df.drop(columns=["original_index_from_matrix_dataframe"])
            self.reduced_dataframe = reduced_df

            # # print information if wanted
            # print("the K genes with the highest variance are:")
            # unflattened_list = row_mapping.iloc[list_of_nlargest_indices].values.tolist()
            # flattened_list_of_nlargest_indices_in_orig_df = [elem[0] for elem in unflattened_list]  # [val for sublist in unflattened_list for val in sublist]
            # temp_df = pd.DataFrame(data=features_dataframe['gene_names'].iloc[flattened_list_of_nlargest_indices_in_orig_df] , columns=['gene_names'])  # .iloc returns values by order. and list_of_nlargest_indices is ordered desecndingly
            # temp_df['variance'] = nlargest_variance_df['variance'].values.tolist() # add a column with the variance values
            # print(temp_df)

        else:    # meaning we are currently creating the TEST DS
            # #save  
            reduced_df = matrix_dataframe.iloc[k_row_indices, :]  # get k rows (genes) with highest variance over all of the columns  
            reduced_df = reduced_df.reset_index()  # this causes a new column to appear - "index" which contains the old indices before resetting
            reduced_df = reduced_df.rename(columns={"index": "original_index_from_matrix_dataframe"})
            self.mapping = reduced_df[["original_index_from_matrix_dataframe"]]
            reduced_df = reduced_df.drop(columns=["original_index_from_matrix_dataframe"])
            self.reduced_dataframe = reduced_df

            # # print information if wanted
            # print("the K genes with the highest variance are:")
            # unflattened_list = row_mapping.iloc[k_row_indices].values.tolist()
            # flattened_list_of_nlargest_indices_in_orig_df = [elem[0] for elem in unflattened_list]  # [val for sublist in unflattened_list for val in sublist]
            # temp_df = pd.DataFrame(data=features_dataframe['gene_names'].iloc[flattened_list_of_nlargest_indices_in_orig_df] , columns=['gene_names'])  # .iloc returns values by order. and list_of_nlargest_indices is ordered desecndingly
            # temp_df['variance'] = nlargest_variance_df['variance'].values.tolist() # add a column with the variance values
            # print(temp_df)





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
                    this practiclay means that we extract a column from the df as our y value

        '''

        if hasattr(self.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            curr_filename = self.imageFolder.samples[index][0]
            curr_img_tensor = self.imageFolder[index][0]  # note that this calls __get_item__ and returns the tensor value
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            curr_img_tensor, curr_filename = self.imageFolder[index]

        # for me
        X = curr_img_tensor  # this is actually X_i

        # get the sample's name from its absolute path and file name
        curr_sample_name = curr_filename.partition('_')[0].partition('/images/')[2]  # first partition to get everything before the first _ , second partition to get everything after /images/

        # get the y value's COLUMN in the gene expression matrix df (with help from the barcodes df)
        index_in_barcoes_df = self.barcodes_dataframe.index[self.barcodes_dataframe['barcodes'] == curr_sample_name].item() # assumption: only 1 item is returned
        column = self.column_mapping.index[self.column_mapping['original_index_from_matrix_dataframe'] == index_in_barcoes_df].item() # assumption: only one item is returned

        # note that unlike before, we want to get the y value of the ENTIRE COLUMN of the REDUCED dataframe (reduced_df should be  K x 4992)
        # and so, return the entire column
        current_gene_expression_values = self.reduced_dataframe.iloc[:, column] 

        # for me
        y = current_gene_expression_values
        # convert to a tensor
        y = torch.from_numpy(y.to_numpy()).float()  # assumption: currently "y" is a pandas series (\dataframe) slice, and needs to be converted to a tensor for later usage
        assert (len(y) == self.num_of_dims_k)

        return X, y, column  # note that "column" is here for future reference, and is the column in matrix_dataframe that these y values belong to


class STDL_Dataset_KValuesPerImg_LatentTensor_NMF(torch.utils.data.Dataset):
    '''
    NOTE: every element of the dataset is a 2d tuple of: (img tensor, k-dim tensor)  ** the tensor is from k-dim latent space
    '''

    def __init__(self, imageFolder, matrix_dataframe, features_dataframe, barcodes_dataframe, column_mapping, num_of_dims_k):
        print("\n----- entering __init__ phase of  STDL_Dataset_KValuesPerImg_LatentTensor_NMF -----")

        # Save important information from outside
        self.imageFolder = imageFolder
        self.matrix_dataframe, self.features_dataframe, self.barcodes_dataframe = matrix_dataframe, features_dataframe, barcodes_dataframe
        self.column_mapping = column_mapping
        # NOTE: the matrix_dataframe above is a reduced version of the original matrix_dataframe after cutting features by condition
        self.num_of_dims_k = num_of_dims_k

        # save additional information
        self.num_of_features_matrix_df = len(matrix_dataframe.index) 
        self.num_of_samples_matrix_df = len(matrix_dataframe.columns)
        self.size_of_dataset = len(self.imageFolder) # NOTE: size_of_dataset != num_of_samples  
        # 290920 save for later use
        if hasattr(self.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            self.num_of_images_with_no_augmentation = self.size_of_dataset
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            self.num_of_images_with_no_augmentation = imageFolder.dataset_lengths_list[0] # NOTE: the concatanated dataset has the original list of datasets inside it. first in that list is the original untransformed imageFolder DS

        # create the reduced dataframe:
        print("performing NMF decomposition on main matrix dataframe ...")
        self.nmf_model = NMF(n_components=num_of_dims_k, init='random', random_state=0)  # TODO: is this the best init type ?
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

        # get the sample's name from its absolute path and file name
        curr_sample_name = curr_filename.partition('_')[0].partition('/images/')[2]  # first partition to get everything before the first _ , second partition to get everything after /images/

        # get the y value's COLUMN in the gene expression matrix df (with help from the barcodes df)
        index_in_barcoes_df = self.barcodes_dataframe.index[self.barcodes_dataframe['barcodes'] == curr_sample_name].item() # assumption: only 1 item is returned
        column = self.column_mapping.index[self.column_mapping['original_index_from_matrix_dataframe'] == index_in_barcoes_df].item() # assumption: only one item is returned

        # note that unlike before, we want to get the y value of the ENTIRE COLUMN of the REDUCED dataframe (reduced_df should be  K x 4992)
        # and so, return the entire column

        current_k_dim_vector = self.reduced_dataframe.iloc[:, column] 

        # for me
        y = current_k_dim_vector
        # convert to a tensor
        y = torch.from_numpy(y.to_numpy()).float()  # assumption: currently "y" is a pandas series (\dataframe) slice, and needs to be converted to a tensor for later usage
        assert (len(y) == self.num_of_dims_k)

        return X, y, column  # note that "column" is here for future reference, and is the column in matrix_dataframe that these y values belong to


class STDL_Dataset_KValuesPerImg_LatentTensor_AutoEncoder(torch.utils.data.Dataset):
    '''
    NOTE: every element of the dataset is a 2d tuple of: (img tensor, k-dim tensor)  ** the tensor is from k-dim latent space
    '''

    def __init__(self, imageFolder, matrix_dataframe, features_dataframe, barcodes_dataframe, AEnet, column_mapping, num_of_dims_k, device):
        print("\n----- entering __init__ phase of  STDL_Dataset_KValuesPerImg_LatentTensor_AutoEncoder -----")

        # Save important information from outside
        self.imageFolder = imageFolder
        self.matrix_dataframe, self.features_dataframe, self.barcodes_dataframe = matrix_dataframe, features_dataframe, barcodes_dataframe
        self.column_mapping = column_mapping
        # NOTE: the matrix_dataframe above is a reduced version of the original matrix_dataframe
        self.num_of_dims_k = num_of_dims_k

        # save additional information
        self.num_of_features_matrix_df = len(matrix_dataframe.index) 
        self.num_of_samples_matrix_df = len(matrix_dataframe.columns)
        self.size_of_dataset = len(self.imageFolder) # NOTE: size_of_dataset != num_of_samples  
        # 290920 save for later use
        if hasattr(self.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            self.num_of_images_with_no_augmentation = self.size_of_dataset
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            self.num_of_images_with_no_augmentation = imageFolder.dataset_lengths_list[0] # NOTE: the concatanated dataset has the original list of datasets inside it. first in that list is the original untransformed imageFolder DS

        # save a torch.Dataset version of our pandas matrix_dataframe
        self.dataset_from_matrix_df = STDL_Dataset_matrix_df_for_AE_init(self.matrix_dataframe)

        # initialize the autoencoder
        print("initializing the autoencoder (this might take a while) ...")
        num_of_rows_in_matrix_df = len(matrix_dataframe.index)
        num_of_features = num_of_rows_in_matrix_df
        num_of_columns_in_matrix_df = len(matrix_dataframe.columns)
        
        self.autoEncoder = AEnet

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

        # get the sample's name from its absolute path and file name
        curr_sample_name = curr_filename.partition('_')[0].partition('/images/')[2]  # first partition to get everything before the first _ , second partition to get everything after /images/

        # get the y value's COLUMN in the gene expression matrix df (with help from the barcodes df)
        index_in_barcoes_df = self.barcodes_dataframe.index[self.barcodes_dataframe['barcodes'] == curr_sample_name].item() # assumption: only 1 item is returned
        column = self.column_mapping.index[self.column_mapping['original_index_from_matrix_dataframe'] == index_in_barcoes_df].item() # assumption: only one item is returned

        # note that unlike before, we want to get the y value of the ENTIRE COLUMN of the REDUCED dataframe (reduced_df should be  K x 4992)
        # and so, return the entire column
        vector_with_all_features = self.matrix_dataframe.iloc[:, column].to_numpy()  
        # convert it to torch
        vector_with_all_features = torch.from_numpy(vector_with_all_features).float().cuda()  #NOTE: note the cuda here !!!
        # encode the vector == convert to a z dimensional latent vector
        latent_vector = self.autoEncoder.encodeWrapper(vector_with_all_features)   

        # for me
        y = latent_vector.squeeze()  # note the squeeze ! this is relevant because the batch size  of "latent_vector" is 1
        assert (len(y) == self.num_of_dims_k)

        return X, y, column  # note that "column" is here for future reference, and is the column in matrix_dataframe that these y values belong to


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

    ## TODO - get from "scalefactors_json" file. (leon's note)
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
    

