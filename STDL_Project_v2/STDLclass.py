import projectLoadAndPreProcess
from projectUtilities import compare_matrices, plot_SingleGene_PredAndTrue_ScatterComparison, plot_SingleGene_PredAndTrue_ColorVisualisation_Mandalay
from projectTrainAndPredict import getSingleDimPrediction, train_prediction_model
from projectModels import get_model_by_name_Mandalay

import os  # check  the import below
from os import walk
import pandas as pd
import numpy as np
import torch
import pathlib

'''
TODO: PLEASE READ THE ASSUMPTIONS BELOW !
'''


class STDLObject(object):

    def __init__(self, path_to_all_data_folders_dir : str, chosen_gene_name : str):
        '''
        init method for the class.
        THIS METHOD PERFORMS THE CREATION AND PRE-PROCESSING OF THE TRAIN AND TEST DATASETS.
        After the init method is performed, we can call train and test on the pre created datasets.
        TODO: Note !! 
        the program is not yet fit to get new data points "on the fly"

        assumption: in the `path_to_all_data_folders_dir` there are many data folders.

        assumption: each of the above described data folders contains:
                        1. a large uncut biopsy image
                        2. a "spots" file (x,y location of data points in the large image)
                        3. a "stdata" file
                        4. ** if pre processing occured already - contains another inner folder `images` of the smaller cut images from the biopsy

        assumption: there is a data folder called 'patient2' which will become the test folder

        assumption: checked genes:
                        1. CRISP3 aka ENSG00000096006 (original)
                        2. MKI67 aka ENSG00000148773
                        3. FOXA1 aka ENSG00000129514
                        
                        ** this does not mean that other genes wont work well, these are just the ones we checked

        steps of the init method:
        - get the working device
        - create a `list_of_training_datasets` from the given path (`path_to_all_data_folders_dir`) [dont forget to read assumptions above]
        - 

        '''
        ###------------------------------------------------------------------------------------------
        # get the working device
        device = self._getAndPrintDeviceData_CUDAorCPU()

        # arrange a subdir list for iteration later on
        subdir_list = [subdir for root, subdir, files in walk(path_to_all_data_folders_dir, topdown=True)][0]
        subdir_list.sort()
        print(f'\nenumerating the subdir_list:\n{[item for item in enumerate(subdir_list)]}\n')
        
        ### init parameters before creating the dataset
        im_hight_and_width_size = 176  
        list_of_training_datasets = []
        test_ds = None
        number_of_folders_without_the_chosen_gene = 0
        max_normalized_stdata_value = -1

        ###------------------------------------------------------------------------------------------
        ### create the `list_of_training_datasets`
        for idx, subdir in enumerate(subdir_list):
            print(f'** creating dataset for biopsy: {subdir}, indexed: {idx+1}/{len(subdir_list)} **')

            current_dir = path_to_all_data_folders_dir + subdir # assumption: there's a "/" between the 2 concatanated strings
            # If the current directory is reserved for test data - then we want it's dataset to be without augmentation
            #      the subdir that is reserved for test data will be denoted 'patient2'
            curr_img_folder = None
            temp_img_folder = None  # created to insert in `cut_samples_with_no_matching_image_and_reorder_df_mandalay` - this should always be the unaugmented version
            if subdir == 'patient2':
                curr_img_folder = projectLoadAndPreProcess.load_dataset_from_images_folder(current_dir + '/images', im_hight_and_width_size)
                temp_img_folder = curr_img_folder
            else:  # meanning this is a train ds
                curr_img_folder = projectLoadAndPreProcess.load_augmented_dataset_from_images_folder(current_dir + '/images', im_hight_and_width_size)
                temp_img_folder = projectLoadAndPreProcess.load_dataset_from_images_folder(current_dir + '/images', im_hight_and_width_size)

            stdata_dataframe = pd.read_csv(current_dir + '/stdata.tsv', sep='\t', index_col=0) #Note: index_col=0 makes the first column the index column #TODO: important !

            # chosen gene verification:
            # if the chosen gene is not in the given stdata dataframe, we want to notify the user but keep on going.
            # for example, the gene CRISP3 does not appear in ~28 / 70 folders from our given datasets (all of them are from the mandalay data)
            if chosen_gene_name not in stdata_dataframe.columns:
                print(f'!!!**** Problem: chosen gene not found in the stdata file of the subdir {subdir} ****!!!\n\n')
                number_of_folders_without_the_chosen_gene += 1
                continue

            # perform cut_samples_with_no_matching_image_and_reorder_df_mandalay
            stdata_dataframe = projectLoadAndPreProcess.cut_samples_with_no_matching_image_and_reorder_df_mandalay(stdata_dataframe, temp_img_folder)

            # normalize the stdata file's values
            stdata_dataframe = projectLoadAndPreProcess.perform_log_1p_normalization(stdata_dataframe)

            # create the current train ds
            curr_train_ds = projectLoadAndPreProcess.STDL_Dataset_SingleValuePerImg_Mandalay(imageFolder=curr_img_folder, 
                                                                            stdata_dataframe=stdata_dataframe, 
                                                                            chosen_gene_name=chosen_gene_name)

            if subdir == 'patient2':
                test_ds = curr_train_ds   # NOTE: this is because "patient2" is our test data !
            else:  # meanning this is a train ds
                # insert to the list of training datasets
                list_of_training_datasets.append(curr_train_ds)

            # print additional info about the current dataset
            curr_max_value = stdata_dataframe.max().max()
            print(f'\nmin value in matrix_dataframe {stdata_dataframe.min().min()} max value in matrix_dataframe {curr_max_value}')
            print(f'\nmedian value in matrix_dataframe {np.median(stdata_dataframe.values)} mean value in matrix_dataframe {np.mean(stdata_dataframe.values)}\n')
            if curr_max_value > max_normalized_stdata_value:
                max_normalized_stdata_value = curr_max_value

            # end of for loop
            pass

        ###------------------------------------------------------------------------------------------
        print(f'finished creating the training datasets loop')
        print(f'its length is: {len(list_of_training_datasets)}')
        print(f'number of folders that did not have the chosen gene in their data table is: {number_of_folders_without_the_chosen_gene}')
        ###------------------------------------------------------------------------------------------

        # save needed variables for later in the STDL class object
        self._combined_ds_train = projectLoadAndPreProcess.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay(list_of_datasets=list_of_training_datasets)
        self._ds_test = test_ds
        self._path_to_all_data_folders_dir = path_to_all_data_folders_dir
        self._device = device
        self._chosen_gene_name = chosen_gene_name
        self._trained_models = {'BasicConvNet': None, 'DenseNet121': None}  # might add more later
        self._max_normalized_stdata_value = max_normalized_stdata_value

        # create folders for output
        self._createOutputFolders()
    

    def trainModel(self, name_of_model_to_train : str, hyperparameters : dict = None):
        '''
        This method trains a model, and saves it to the class object.

        if no hyperparameters are given, some default ones are created.

        TODO: Note !!
        currently, only 2 types of models are supported: 
        1. BasicConvNet
        2. DenseNet121
        if this method recieves any other `name_of_model_to_train` an exception will be raised

        '''
        ## create hyperparameters if they werent recieved as an argument 
        if hyperparameters is None:        
            # define hyperparameters for the TRAINING of the models (NOT the testing phases of the experiments)
            hyperparameters = dict()
            hyperparameters['batch_size'] = 30
            hyperparameters['max_alowed_number_of_batches'] = 99999 #<--------------------------change to inf or 99999. anything below 1220 will cut some batches ... this is only used to speed up training
            hyperparameters['precent_of_dataset_allocated_for_training'] = 0.8  # TODO currently not used
            hyperparameters['learning_rate'] = 1e-4
            hyperparameters['momentum'] = 0.9
            hyperparameters['num_of_epochs'] = 3 #<------------------------------------------change to 5 at least
            hyperparameters['num_workers'] = 2 #<------------------------------------------ NOTE: default is 0, means everything happens serially. testing 2 now !
                                            # see: https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading

            # define hyperparameters for BasicConvNet
            hyperparameters['channels'] = [32, 32, 64, 64] 
            hyperparameters['num_of_convolution_layers'] = len(hyperparameters['channels'])
            hyperparameters['hidden_dims'] = [100, 100]
            hyperparameters['num_of_hidden_layers'] = len(hyperparameters['hidden_dims'])
            hyperparameters['pool_every'] = 99999

            # add the chosen single gene's index to the hyperparameters
            hyperparameters['gene_name'] = self._chosen_gene_name

        # whether or not we got it externally - 
        # save the hyperparameters to the stdl object
        self._hyperparams = hyperparameters



        ## create the model
        model = get_model_by_name_Mandalay(name=name_of_model_to_train, dataset=self._combined_ds_train.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay[-1], hyperparams=self._hyperparams)
        ## create the loss function and optimizer
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
        ## train the model
        trained_model = train_prediction_model(model_to_train=model, ds_train=self._combined_ds_train, loss_fn=loss_fn, 
                                                optimizer=optimizer, hyperparams=hyperparameters, 
                                                model_name=name_of_model_to_train, dataset_name='single_gene_Mandalay', device=self._device)
                                                # TODO: might change later: dataset_name='single_gene_Mandalay'

        ## update and save the information
        if name_of_model_to_train == 'BasicConvNet':
            self._trained_models['BasicConvNet'] = trained_model
        elif name_of_model_to_train == 'DenseNet121':
            self._trained_models['DenseNet121'] = trained_model
        else: 
            raise Exception(f'requested model type is not yet supported ....')


    def runModelOnTestData(self, name_of_model_to_run : str, hyperparams: dict = None):
        '''
        This function runs the pretrained requested model on the test data which was determined in the init phase of the STDL object

        there are 2 testing steps: 
        1. check our trained model on items from the train dataset - this is performed to verify that the model actually learned something
        2. check our trained model on test data

        for each of the above steps we perform:
        1. compare_matrices - this will compare the truth value with the prediction values.
        2. plot_SingleGene_PredAndTrue_ScatterComparison - this plots and saves a scatter plot with a comparison between  truth and predicted values
        3. plot_SingleGene_PredAndTrue_ColorVisualisation_Mandalay - this creates colored plots created from the truth and prediction values given. these can later be compared with the actuall large biopsy images

        if the hyperparams arent given, the ones used for the training are also used here.

        if the `name_of_model_to_run` does not exist, an exception will be raised

        if the `name_of_model_to_run` does exist, but was not trained yet, an exception will be raised

        '''
        # get the requested model
        model = None
        if name_of_model_to_run == 'BasicConvNet':
            model = self._trained_models['BasicConvNet']
        elif name_of_model_to_run == 'DenseNet121':
            model = self._trained_models['DenseNet121']
        else:
            raise Exception("model name does not exist ....")
        
        if model is None:
            raise Exception("model was not yet trained ....")
        
        # get the relevant hyperparams
        if hyperparams is None: 
            hyperparams = self._hyperparams

        # temp line to deal with later when there will be more functionality
        dataset_name = 'single_gene_Mandalay'
        model_name = name_of_model_to_run

        #####
        # perform on TRAIN data  (this is done to validate that we actually learned something)
        print("\n## perform on TRAIN data ##")
        M_truth, M_pred = getSingleDimPrediction(dataset=self._combined_ds_train.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay[-1], model=model, device=self._device)  # TODO: Note!!!: the last item in ds_train_list is supposed to be "patient1" and thats why `dataset=ds_train_list[-1]`
        baseline = np.full(shape=M_truth.shape, fill_value=np.average(M_truth))  # `full` creates an array of wanted size where all values are the same fill value
        compare_matrices(M_truth, M_pred, Baseline=baseline)
        plot_SingleGene_PredAndTrue_ScatterComparison(self._combined_ds_train.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay[-1], M_pred, M_truth, model_name, dataset_name + ' Train', hyperparams['gene_name'])  # the combined DS isnt used, thats why its ok to send it there
        # plot_SingleGene_PredAndTrue_ColorVisualisation_Mandalay(combined_ds_train, M_pred, M_truth, model_name, dataset_name + ' Train', hyperparams['gene_name'])
        plot_SingleGene_PredAndTrue_ColorVisualisation_Mandalay(self._combined_ds_train.STDL_ConcatDataset_of_SingleValuePerImg_Datasets_Mandalay[-1], M_pred, M_truth, model_name, dataset_name + ' Train', hyperparams['gene_name'])
        

        # perform on TEST data
        print("\n## perform on TEST data ##")
        M_truth_test, M_pred_test = getSingleDimPrediction(dataset=self._ds_test, model=model, device=device)
        baseline = np.full(shape=M_truth_test.shape, fill_value=np.average(M_truth))  #NOTE: shape of TEST data, filled with TRAIN data avg !!! # `full` creates an array of wanted size where all values are the same fill value
        compare_matrices(M_truth_test, M_pred_test, Baseline=baseline)  #NOTE: same baseline as above - the TRAIN baseline
        plot_SingleGene_PredAndTrue_ScatterComparison(self._ds_test, M_pred_test, M_truth_test, model_name, dataset_name + ' Test', hyperparams['gene_name'])  # the combined DS isnt used, thats why its ok to send it there
        # plot_SingleGene_PredAndTrue_ColorVisualisation_Mandalay(ds_test, M_pred_test, M_truth_test, model_name, dataset_name + ' Test', hyperparams['gene_name'])
        plot_SingleGene_PredAndTrue_ColorVisualisation_Mandalay(self._ds_test, M_pred_test, M_truth_test, model_name, dataset_name + ' Test', hyperparams['gene_name'])

        
    def getModelTrainingStatus(self):
        '''
        This is a short and simple helper function that simply prints the training status
        of the different existing models.
        currently only supports
        1. BasicConvNet
        2. DenseNet121
        '''
        
        print(f'model training status:')
        for key in self._trained_models.keys():
            if self._trained_models[key] is None:
                print(f'{key} status: NOT trained')
            else:
                print(f'{key} status: trained') 

        print(f'-- end of model training status --\n\n')


    def preProcess_prepareDataFolders(self):
        '''
        iterate over the given data folders - for each make sure that 
            1. the stdata file is good
            2. the spots file is good
            3. there exists a large image
            4. if there is no `images` folder with little cut images - then create them
        '''
        pass  # TODO: NOT YET IMPLEMENTED. need some rearranging from other EXISTING functions
              #         please see relevant functions in projectLoadAndPreProcess.py


    def _getAndPrintDeviceData_CUDAorCPU(self):
        '''
        This function is supposed to be used only once during the STDL object's init phase.
        it returns a cuda object if it exists, and if not, returns a cpu device.
        also, the function prints information on the current gpu if it exists.
        '''
        #
        print(f'\nPerforming: getAndPrintDeviceData_CUDAorCPU')
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  
        print(f'cuda debugging allowed')
        #
        print(f'cuda device count: {torch.cuda.device_count()}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        #Additional Info when using cuda
        if device.type == 'cuda':
            print(f'device name: {torch.cuda.get_device_name(0)}')
            print(f'torch.cuda.device(0): {torch.cuda.device(0)}')
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        # NOTE: important !!!!!!
        # clearing out the cache before beginning
        torch.cuda.empty_cache()
        print(f'\nfinished: getAndPrintDeviceData_CUDAorCPU')
        return device


    def _createOutputFolders(self):
        '''
        This function is supposed to be used only once during the STDL object's init phase.
        its purpose is to make sure that we have the needed output folders so we can save plots later on
        currently there are 4 directories as you can see below:
        1. color visualization dir
        2. heatmaps dir
        3. loss convergences dir
        4. scatter plot dirs
        '''
        current_dir = str(pathlib.Path(__file__).parent)
        # create folders if they do not exist
        pathlib.Path(current_dir + '/saved_plots_color_visualisation').mkdir(parents=True, exist_ok=True)
        pathlib.Path(current_dir + '/saved_plots_heatmaps').mkdir(parents=True, exist_ok=True)
        pathlib.Path(current_dir + '/saved_plots_loss_convergence').mkdir(parents=True, exist_ok=True)
        pathlib.Path(current_dir + '/saved_plots_scatter_comparisons').mkdir(parents=True, exist_ok=True)
