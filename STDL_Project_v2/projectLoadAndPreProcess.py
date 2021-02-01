import csv
import gzip
import os
import cv2
from tqdm import tqdm
from time import sleep
import shutil

import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import numpy as np

import projectModels
import projectUtilities 

import torch
import torchvision
import torchvision.transforms as torchTransform
from torchvision.datasets import ImageFolder, DatasetFolder
from sklearn.decomposition import NMF
from torch.utils.data import Dataset, DataLoader, ConcatDataset


## -------------------------------------------------------------------------
##
##  Below: functions used to assist preparaing our data for insertion and usage 
##          inside the STDL classes that appear in this file
##
## -------------------------------------------------------------------------


def load_dataset_from_images_folder(path_to_images, im_hight_and_width_size):
    '''
    `load_dataset_from_images_folder`
    This function creates a pytorch `ImageFolder` object (which is a dataset) from the given image folder path

    NOTE: the dataset refered to here is the imageFolder dataset created from the original images folder
    NOTE: edge case in which the folder path is not correct or does not exist was not implemented.
    '''
    print("\n----- entered function load_dataset_from_pictures_folder -----")

    tf = torchTransform.Compose([
        # Resize to constant spatial dimensions
        torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
        # Convert image to greyscale
        torchTransform.Grayscale(num_output_channels=3), # 3 means: R==G==B. this is important for the model inputs later
        # PIL.Image -> torch.Tensor
        torchTransform.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        torchTransform.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    dataset_object = ImageFolder(os.path.dirname(path_to_images), tf)

    print("\n----- finished function load_dataset_from_pictures_folder -----\n")

    return dataset_object


def load_augmented_dataset_from_images_folder(path_to_images, im_hight_and_width_size):
    '''
    `load_augmented_dataset_from_images_folder`
    This function creates a pytorch `ImageFolder` object (which is a dataset) from the given image folder path
    This is done by creating several dataset object, and then concatenating them using the `STDL_ConcatDataset_of_ImageFolders` class
    which can be seen below in this file
    the function is very similar to the above `load_dataset_from_images_folder` only with more transformation for the augmentation

    Augmentation: the augmentation of the dataset is expressed in the large number of transformations.
                    every image is `transformed` 8 times:
                        normal, 90^, 180^, 270^, flipped, flipped 90^, flipped 180^, flipped 270^

    the code below creats a new `ImageFolder` dataset object after each transformation. everntuall they will all be concatanated as mentioned above.

    NOTE: the dataset refered to here is the imageFolder dataset created from the original images folder
    NOTE: edge case in which the folder path is not correct or does not exist was not implemented.
    NOTE: dont freak out from the large amount of code, most of this function is transformations that are "copy-pasted" with slight differences.
          There are overall 8 differnt transformation - 0/90/180/270 , and the same four flipped horizontaly
    '''
    print("\n----- entered function load_augmented_dataset_from_images_folder -----")
    
    # note that this next "compose" actually a pipeline
    tf_original =   torchTransform.Compose([
                    # Resize to constant spatial dimensions
                    torchTransform.Resize((im_hight_and_width_size, im_hight_and_width_size)),
                    # Convert image to greyscale
                    torchTransform.Grayscale(num_output_channels=3), # 3 means: R==G==B. this is important for the model inputs later,
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
                    # Convert image to greyscale
                    torchTransform.Grayscale(num_output_channels=3), # 3 means: R==G==B. this is important for the model inputs later,
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
                    # Convert image to greyscale
                    torchTransform.Grayscale(num_output_channels=3), # 3 means: R==G==B. this is important for the model inputs later,
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
                    # Convert image to greyscale
                    torchTransform.Grayscale(num_output_channels=3), # 3 means: R==G==B. this is important for the model inputs later,
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
                    # Convert image to greyscale
                    torchTransform.Grayscale(num_output_channels=3), # 3 means: R==G==B. this is important for the model inputs later,
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
                    # Convert image to greyscale
                    torchTransform.Grayscale(num_output_channels=3), # 3 means: R==G==B. this is important for the model inputs later,
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
                    # Convert image to greyscale
                    torchTransform.Grayscale(num_output_channels=3), # 3 means: R==G==B. this is important for the model inputs later,
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
                    # Convert image to greyscale
                    torchTransform.Grayscale(num_output_channels=3), # 3 means: R==G==B. this is important for the model inputs later,
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


    print("\n----- finished function load_augmented_dataset_from_images_folder -----\n")

    return final_dataset_object


def perform_log_1p_normalization(df):
    '''
    perform log 1P normaliztion on the dataframe matrix values:
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


def cut_samples_with_no_matching_image_and_reorder_df_mandalay(stdata_df, image_folder_of_the_df):
    '''
    Goal: cut samples that do not have matching images

    Challenge: samples might be missing from both ends - samples from stdata_df might not be present in the image folder and vice verse.

    This function solves this challenge to obtain the goal.

    NOTE: this function also reorders the dataframe. (see the note in the end of this function)

    '''
    print(f'cutting samples that dont have mathching images in the image folder from the dataframe ...')

    # verify that this is a regular (and not augmented) image folder
    if not hasattr(image_folder_of_the_df, 'samples'):  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
        raise NameError(' wrong image folder type... insert the regular, not augmented one ')
    
    #
    list_of_index_tuples = [] # each element in the list will be a tuple containing (index in image folder, column index in the orig df, column index in the new df)    

    # get indices of samples that DO exist in the image folder, add them to `existing_sampexisting_samples_list_in_image_folderles_list`
    existing_samples_list_in_image_folder = []
    for index_in_image_folder, element in enumerate(image_folder_of_the_df.samples):
        filename = element[0]
        curr_sample_name = filename.partition('_x')[0].partition('/images/')[2].partition('_')[0]  # [0] means verything before the token, [2]  means everything after the token
        existing_samples_list_in_image_folder.append(curr_sample_name)
        
    existing_samples_list_in_stdata_df = stdata_df.index.to_list()

    existing_samples_list_intersection = list(set(existing_samples_list_in_image_folder) & set(existing_samples_list_in_stdata_df))

    existing_samples_list_intersection.sort() # TODO: check if needed - see code line below for reason to sort in the first place

    # save the updated dataframe  
    updated_df = stdata_df.loc[existing_samples_list_intersection,:] # keep only the wanted rows from the dataframe # NOTE !!!!! this line returns a dataframe in which the rows are ORDERED BY THE ORDER OF `column_list` !!!!
    # print(f'original df info:\n{stdata_df.info()}')  #TODO: delete later
    # print(f'updated_df info:\n{updated_df.info()}')  #TODO: delete later   

    print(f'V   done :)\n')
    return updated_df


## -------------------------------------------------------------------------
##
##  End of the segment above 
##  Below: classes used to load and maintain images and stdata information
##         These are the actually stdata datasets.
##
## -------------------------------------------------------------------------


class STDL_Dataset_SingleValuePerImg_Mandalay(torch.utils.data.Dataset):
    '''
    `STDL_Dataset_SingleValuePerImg_Mandalay`
    this is the main custom dataset class that will hold information on images and gene expression value.
    NOTE: every element of the dataset is a 2d tuple of: (img tensor, gene exp value)
    NOTE: the above gene exp value is for a given specific gene
    NOTE: this class by its nature uses 'lazy allocation' - when initializing the dataset, nothing is actually being loaded but addresses and links.
            only when invoking `__getitem__`with a specific index -  a single image is attached to its gene expression level
    '''

    def __init__(self, imageFolder, stdata_dataframe, chosen_gene_name):
        print("\n----- entering __init__ phase of  STDL_Dataset_SingleValuePerImg -----")

        # Save important information from outside
        self.imageFolder = imageFolder
        self.stdata_dataframe = stdata_dataframe
        self.gene_name = chosen_gene_name
        # self.row_mapping, self.column_mapping = row_mapping, column_mapping #TODO: not sure if needed

        # save additional information
        self.num_of_features_stdata_df = len(stdata_dataframe.columns)        #TODO: verify: does this include the headers or not !?!?
        self.num_of_samples_stdata_df = len(stdata_dataframe.index)       #TODO: verify: does this include the headers or not !?!?
        self.size_of_dataset = len(self.imageFolder) # NOTE: size_of_dataset != num_of_samples  when the dataset is AUGMENTED - see if clause below
        #  
        if hasattr(self.imageFolder, 'samples'):  # meaning this is a regular "ImageFolder" type
            self.num_of_images_with_no_augmentation = self.size_of_dataset
        else:  # meaning this is a custom DS I built - STDL_ConcatDataset_of_ImageFolders
            self.num_of_images_with_no_augmentation = imageFolder.dataset_lengths_list[0] # NOTE: the concatanated dataset has the original list of datasets inside it. first in that list is the original untransformed imageFolder DS

        '''
        create the reduced dataframe == a dataframe with only one row
        '''
        # first get the requested gene's column index + check if requested gene name actually appears in the stdata_dataframe
        requested_column_index = -1
        if chosen_gene_name not in stdata_dataframe.columns:
            raise ValueError('A very specific bad thing happened.')  #TODO: this means that someone who creates this custom dataset needs to do so using TRY CATCH
        else:
            requested_column_index = stdata_dataframe.columns.to_list().index(chosen_gene_name)  #Note: please see: https://stackoverflow.com/questions/176918/finding-the-index-of-an-item-in-a-list # assumption - theres only one occurunce of the gene name in the list

        # get the reduced dataframe
        self.reduced_dataframe = self.stdata_dataframe.iloc[:, requested_column_index]  # get only the relevant gene's COLUMN over ALL samples (== all rows)
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

        # print(f'\ncurr_filename {curr_filename}')  # TODO: TO DELETE !!!<----

        curr_sample_name = curr_filename.partition('_x')[0].partition('/images/')[2].partition('_')[0]  # [0] means verything before the token, [2]  means everything after the token
        
        # print(f'curr_sample_name {curr_sample_name}')  # TODO: TO DELETE !!!<----

        # get the y value's ROW in the gene expression matrix MTX

        # print(f'inside __getitem__ with index {index}, curr_sample_name is {curr_sample_name} curr_filename {curr_filename}')  # TODO: delete later
        current_gene_expression_value = 0
        if curr_sample_name not in self.reduced_dataframe.index.tolist():  # Note: remember that the reduced df here is a pandas series object
            current_gene_expression_value = 0               #TODO: THIS IS "THE DANGEROUS ASSUMPTION" - if the stdata file does not contain information about this image - I assume that its LOG1P normalized value is 0 !!!!!!!!!!!!!
        else:
            current_gene_expression_value = self.reduced_dataframe.at[curr_sample_name]



        # for me
        y = current_gene_expression_value

        return X, y   #, column  #TODO: comment to the left is 171120 #Note: "column" is here for future reference, and is the column in matrix_dataframe that this y value belongs to


class STDL_ConcatDataset_of_ImageFolders(torch.utils.data.Dataset):
    '''
    `STDL_ConcatDataset_of_ImageFolders`
    This is a concatanation of ImageFolder datasets into one unified dataset of images. 
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


class STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay(torch.utils.data.Dataset):
    '''
    `STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay`
    This is a concatanation of `STDL_Dataset_SingleValuePerImg_Mandalay` datasets into one.
    it is needed because for every different "patient" / "biopsy sample" a different `STDL_Dataset_SingleValuePerImg_Mandalay` is created.

    NOTE: every element of the dataset is a 2d tuple of: (img tensor, gene exp value)
    NOTE: the above gene exp value is for a specific gene
    '''

    def __init__(self, list_of_datasets):
        print("\n----- entering __init__ phase of  STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay -----")

        # Save important information from outside
        self.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay = list_of_datasets


        self._list_of_ds_sizes = [len(ds) for ds in list_of_datasets]
        self.index_offsets = np.cumsum(self._list_of_ds_sizes)  # cumsum = cumulative sum. this returns a list (length of datasets_list) in which every element is a cumulative sum of the length that far.
                                                                                              # say the original DS is size 30. then this returns: [30,60,90,120,...]
        
        self._list_of_ds_sizes_with_no_augmentation = [ds.num_of_images_with_no_augmentation for ds in list_of_datasets]
        self.num_of_images_with_no_augmentation =  np.cumsum(self._list_of_ds_sizes_with_no_augmentation)  # cumsum = cumulative sum. this returns a list (length of datasets_list) in which every element is a cumulative sum of the length that far.
                                                                                              # say the original DS is size 30. then this returns: [30,60,90,120,...]
        


        # self._get_item_track_list = self._list_of_ds_sizes      # see logic in __getitem__
        # self._curr_ds_index_to_getitem_from = 0           # see logic in __getitem__

        self._num_of_all_samples = np.sum(self._list_of_ds_sizes)

        print("\n----- finished __init__ phase of  STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay -----\n")

    def __len__(self):
        return self._num_of_all_samples

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
                # return entry from the relevant ds
                return self.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay[dataset_index][final_index_in_ds]
            else: 
                pass
        # if we got here, the index is invalid
        raise IndexError(f'{index} exceeds {self.length}')


        return 

    def _save_images_for_leon(self):
        '''
        This is a temporary function.
        it was used to generate sample images for leon to show the differences between different biopsy samples from 10x genomics and mandalay
        '''
        ds_mandalay = self.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay[-3]
        ds_patient1 = self.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay[-2]
        ds_patient2 = self.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay[-1]

        from torchvision.utils import save_image
        # tensor_to_pil = torchTransform.ToPILImage()(ds_mandalay[10][0].squeeze_(0))
        # save 4 images from ds_mandalay
        save_image(ds_mandalay[10][0], 'leon_mandalay_img1.png')
        save_image(ds_mandalay[11][0], 'leon_mandalay_img2.png')
        save_image(ds_mandalay[12][0], 'leon_mandalay_img3.png')
        save_image(ds_mandalay[20][0], 'leon_mandalay_img4.png')
        # save 4 images from ds_patient1
        save_image(ds_patient1[10][0], 'leon_patient1_img1.png')
        save_image(ds_patient1[11][0], 'leon_patient1_img2.png')
        save_image(ds_patient1[12][0], 'leon_patient1_img3.png')
        save_image(ds_patient1[20][0], 'leon_patient1_img4.png')
        # save 4 images from ds_patient2
        save_image(ds_patient2[11][0], 'leon_patient2_img1.png')
        save_image(ds_patient2[12][0], 'leon_patient2_img2.png')
        save_image(ds_patient2[13][0], 'leon_patient2_img3.png')
        save_image(ds_patient2[20][0], 'leon_patient2_img4.png')
        

## -------------------------------------------------------------------------
##
##  End of the segment above 
##  Below: functions used to prepare data from mandalay in usable folders.
##
## -------------------------------------------------------------------------
'''
IMPORTANT !

The order of the functions below is crucial:

0. the assumption is that you download all your data from scratch. if you already have what was previously used, this is not needed.
1. Download data from mandalay and 10x genomics
2. use `create_stdata_file_from_mtx` to transform the stdata files from the 10x genomics version (they have 3 files instead of 1: barcodes, features, matrix) into the mandalay version (only 1 file: stdata)
3. use `create_folders_from_new_mandalay_data` on disarray data from mandaly. this creates a folder, with subfolders for each biopsy from mandalay
4. use `create_image_subfolders_in_new_mandalay_data_folders` to create "/images" sub folders iniside biopsy folders mentioned in `3.`
5. `create_image_subfolders_in_new_mandalay_data_folders` calls `create_image_subfolders_in_new_mandalay_data_folders` for every large biopsy image. 
    this in turn will cut the large biopsy image into smaller ones using the "spots" files.
'''

def create_folders_from_new_mandalay_data(path_to_dir):
    '''
    `create_folders_from_new_mandalay_data`
    Function that was used ONE TIME ONLY to create folders from the data downloaded from mandalay.

    NOTE: the assumption is that all files are in dissarray in a single folder with a given path.
    NOTE: there are only 3 file types in the folder: csv (for spots), jpg (biopsy images that will later be cut), tsv (for the stdata)
    NOTE: there are many prints. they were used to help see that the process was working correctly, and are not actually needed
    '''

    spots_filenames = [filename for filename in listdir(path_to_dir) if filename.endswith(".csv") and not filename.__contains__("metadata")]
    images_filenames = [filename for filename in listdir(path_to_dir) if filename.endswith(".jpg")]
    stdata_filenames = [filename for filename in listdir(path_to_dir) if filename.endswith(".tsv")]
    spots_filenames.sort()
    images_filenames.sort()
    stdata_filenames.sort()

    print(spots_filenames)
    print(f'****')
    print(images_filenames)
    print(f'****')
    print(stdata_filenames)
    print(f'****')

    ## testing
    ### curr_sample_name = curr_filename.partition('_')[0].partition('/images/')[2]  # first partition to get everything before the first _ , second partition to get everything after /images/
    sample_names_from_spots_filenames = [(name.partition('spots_')[2].partition('.')[0])[2:] for name in spots_filenames]
    print(sample_names_from_spots_filenames)
    print(f'****')
    sample_names_from_images_filenames = [(name.partition('HE_')[2].partition('.')[0])[2:] for name in images_filenames]
    print(sample_names_from_images_filenames)
    print(f'****')
    sample_names_from_stdata_filenames = [(name.partition('_stdata')[0])[2:] for name in stdata_filenames]
    print(sample_names_from_stdata_filenames)
    print(f'****')

    print(f'lengths: {len(sample_names_from_spots_filenames)}, {len(sample_names_from_images_filenames)}, {len(sample_names_from_stdata_filenames)}')
    intersection = [name for name in sample_names_from_images_filenames if name in sample_names_from_spots_filenames and name in sample_names_from_stdata_filenames]
    print(f'intersection length {len(intersection)} intersection:\n{intersection}')
    print(f'****')

    ## now that we know that the intersection is full:
    new_folder_names = sample_names_from_images_filenames

    

    # create a new folder for every biposy sample, each with 3 files:
    # 1. original_image.jpg
    # 2. spots.csv
    # 3. stdata.tsv
    # NOTE: in a different function, a 4th object is added: an /images folder that will contain all small images cut from the large biopsy image
    for name in new_folder_names:
        # create the new folder name

        dir_name = os.path.join(path_to_dir, name)
        if not os.path.exists(dir_name):
            # os.umask(0)            # give permissions to the folder
            # os.makedirs(dir_name)
            try:
                original_umask = os.umask(0)
                os.makedirs(dir_name, mode=0o777)
            finally:
                os.umask(original_umask)

        ## copy all files into the new directory # TODO: copy them with identical names over all files ???

        image_filename = [s for s in images_filenames if name in s][0]   # return first match that contains "name"
        spots_filename = [s for s in spots_filenames if name in s][0]    # return first match that contains "name"
        stdata_filename = [s for s in stdata_filenames if name in s][0]  # return first match that contains "name"

        shutil.copy2(src=path_to_dir + image_filename, dst=dir_name + "/original_image.jpg")  #TODO: verify endings
        shutil.copy2(src=path_to_dir + spots_filename, dst=dir_name + "/spots.csv")
        shutil.copy2(src=path_to_dir + stdata_filename, dst=dir_name + "/stdata.tsv")


def create_image_subfolders_in_new_mandalay_data_folders(path_to_dir):
    '''
    `create_image_subfolders_in_new_mandalay_data_folders`
    this creates an "/images" folder inside each biposy folder
    it will be used to keep the smaller images cut from the large biposy image using the spots files (this is all invoked
    from `create_smaller_images_from_large_image_in_mandalay_data`)
    '''
    print(f'\n\nentered: create_image_subfolders_in_new_mandalay_data_folders')
    subdir_list = [subdir for root, subdir, files in os.walk(path_to_dir, topdown=True)][0]

    print(subdir_list)

    # create an images folder under every subdir
    for subdir in subdir_list:
        print(subdir)
        create_smaller_images_from_large_image_in_mandalay_data(path_to_dir=path_to_dir + subdir) # assumption: there's a "/" between the 2 concatanated strings


def create_smaller_images_from_large_image_in_mandalay_data(path_to_dir):
    '''
    `create_smaller_images_from_large_image_in_mandalay_data`
    Function to create smaller images from a larger biposy image.
    this is done using the spots file.
    '''
    print("\n----- entered function create_smaller_images_from_large_image_in_mandalay_data -----")
    print(f' given path: {path_to_dir}')
    # data points(x,y coordinate) for the tissue
    path1 = path_to_dir + "/spots.csv"
    positions_dataframe = pd.read_csv(path1)
    positions_dataframe.columns = ['index', 'x', 'y']

    # print(positions_dataframe) # TODO: important ! comment this later
    # print(f'path1:\n {path1}')

    ## import image: comes with BGR
    path2 = path_to_dir + "/original_image.jpg"
    # print(f'path2:\n {path2}')
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
    # NOTE: when "spot_diameter_fullres = 177.482" is given - the output image's size will be 176x176
    spot_diameter_fullres = 177.4829519178534
    spot_radius = int(spot_diameter_fullres/2)

    # num of iterations
    total_amount_of_spots = len(positions_dataframe.index)

    for idx, row in positions_dataframe.iterrows():
        #
        barcode = row['index']
        x = round(row['x'])
        y = round(row['y'])

        #file names
        square_file_name = "{}_x{}_y{}_square.png".format(barcode,x,y)  # previously: '{}_x{}_y{}_{}_square.png'.format(idx,x,y,barcode)

        #print progress
        print(f'processing image {idx + 1} of {total_amount_of_spots} with name: {square_file_name}', end='\r')

        # square image
        roi_square = img[y-spot_radius:y+spot_radius, x-spot_radius:x+spot_radius]
        # plt.imshow(roi_square)
        cv2.imwrite(out_path + square_file_name, roi_square)


    pass
    print(f'\nfinished cutting the big image')
    print("\n----- finished function create_smaller_images_from_biopsy_sample -----")


def create_stdata_file_from_mtx(path_to_10x_genomics_data_dir: str = None):
    '''
    `create_stdata_file_from_mtx`
    This function is used to reformat the stdata files from the 10x genomics version, to the mandalay version
    after this function is done, a new `stdata.tsv` file is created, and the 3 older files: 
    features.tsv, barcodes.tsv, matrix.mtx - are no longer needed.

    NOTE: assumption: the structure of the 10x genomics files:
          main folder -> patient 1 (or 2) folder -> 3 files: features, barcodes, matrix.
    NOTE: the final new stdata file is saved in the same subfolders of the 10x genomics father folder
    '''
    if path_to_10x_genomics_data_dir is None:
        path_patient1_files = "C:/Users/royru/Downloads/new data STDL project from mandalay/patient1"
        path_patient2_files = "C:/Users/royru/Downloads/new data STDL project from mandalay/patient2"
    else: 
        path_patient1_files = path_to_10x_genomics_data + "/patient1"
        path_patient2_files = path_to_10x_genomics_data + "/patient2"

    ####
    for path_to_mtx_tsv_files_dir in [path_patient1_files, path_patient2_files]:

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
        matrix_dataframe = pd.DataFrame.sparse.from_spmatrix(
            matrix)  # NOTE: from: https://pandas.pydata.org/docs/user_guide/sparse.html
        print("V  finished reading matrix.mtx")

        ## dataframe adjusting if needed .....
        # print("adjusting matrix_dataframe")
        # matrix_dataframe = matrix_dataframe.replace([np.inf, -np.inf], np.nan)  # replace all inf values with a NaN value
        # matrix_dataframe = matrix_dataframe.fillna(0)  # fill all NaN values with 0 ....
        # matrix_dataframe = matrix_dataframe.astype(int) #convert value types to int
        # print("V  finished working on matrix_dataframe")

        # update column and row names !
        matrix_dataframe.index = features_dataframe.iloc[:, 0].to_list() # == feature_names_list  # == rows names
        matrix_dataframe.columns = barcodes_dataframe.iloc[:, 0].to_list()

        # print(matrix_dataframe)  # TODO: print to delete
        matrix_dataframe = matrix_dataframe.transpose()  # to fit the new files in the same shape
        # print(matrix_dataframe)  # TODO: print to delete

        # finally, save the new stdata csv file
        matrix_dataframe.to_csv(path_to_mtx_tsv_files_dir + "/stdata.tsv", sep = '\t')

    # end of for loop
    # end of function



