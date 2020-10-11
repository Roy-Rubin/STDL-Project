
import matplotlib.pyplot as plt

def main():
    print("Got to main file. beginning !")
    
    '''
    This contains all the cells from the jupyter notebook
    TODO: Later on, make this more readable by (among other things) dividing this intro smaller functions
    '''

    
    # # **Spatial Transcriptomics Deep Learning (STDL) Project Notebook**
    # 
    # > The notebook contains main experiments and examples of how to use the code

    # ## **Phase 1: Pre-processing and technical preparations**

    # ### 1.1: **Assign GPU device and allow CUDA debugging**


    # the next 2 lines are to allow debugging with CUDA !
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  
    print(f'cuda debugging allowed')


    import torch
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


    # ### 1.2: **Import the Pre-Process module**
    # 
    # > `loadAndPreProcess` module contains methods to load the data files as pytorch and pandas objects, methods to preprocess the given data, and methods to create custom datasets from the preprocessed data.

    # <div class="alert alert-block alert-warning">
    # <b>TODO:</b> fill above line
    # </div>

    


    # note: path to project is: /home/roy.rubin/STDLproject/
    import loadAndPreProcess


    # ### 1.3: **Load pytorch dataset objects from the image folder**
    # 
    # > loading regular and augmented datasets created from the given image folder with transformations.
    # 
    # > Note: `augmentedImageFolder` is a custom dataset of imageFolder objects with different transformations (see code).
    # 
    # > Note: `im_hight_and_width_size` will define the size to which the images in the folder will be resized to. their original size 176, and so if the number will be bigger, the images will be automaticaly upsampled in the `resize` (not sure by what method) - which means images might be "pixelized" / lower quality. The problem is, size 176 doesnt work with all models, so i had to increase the size.

    im_hight_and_width_size = 176  # values: 176 (doesnt work with inception) / 224 (doesnt work with inception) / 299 (works with inception)

    path_to_images_dir_patient1_train = "/home/roy.rubin/STDLproject/spatialGeneExpressionData/patient1/images"
    imageFolder_train = loadAndPreProcess.load_dataset_from_images_folder(path_to_images_dir_patient1_train, im_hight_and_width_size)
    augmentedImageFolder_train = loadAndPreProcess.load_augmented_imageFolder_DS_from_images_folder(path_to_images_dir_patient1_train, im_hight_and_width_size)

    path_to_images_dir_patient2_test = "/home/roy.rubin/STDLproject/spatialGeneExpressionData/patient2/images"
    imageFolder_test = loadAndPreProcess.load_dataset_from_images_folder(path_to_images_dir_patient2_test, im_hight_and_width_size)
    # augmentedImageFolder_test = loadAndPreProcess.load_augmented_imageFolder_DS_from_images_folder(path_to_images_dir_patient2_test, im_hight_and_width_size) # not needed for now


    # ### 1.4: **Load pandas dataframe objects from the given mtx/tsv/csv files**
    # 
    # > `matrix_dataframe` represents the gene expression count values of each sample for each gene
    # 
    # > `features_dataframe` contains the names of all the genes
    # 
    # > `barcodes_dataframe` contains the names of all the samples

    
    path_to_mtx_tsv_files_dir_patient1_train = "/home/roy.rubin/STDLproject/spatialGeneExpressionData/patient1"
    matrix_dataframe_train, features_dataframe_train , barcodes_dataframe_train = loadAndPreProcess.load_dataframes_from_mtx_and_tsv_new(path_to_mtx_tsv_files_dir_patient1_train)

    path_to_mtx_tsv_files_dir_patient2_test = "/home/roy.rubin/STDLproject/spatialGeneExpressionData/patient2"
    matrix_dataframe_test, features_dataframe_test , barcodes_dataframe_test = loadAndPreProcess.load_dataframes_from_mtx_and_tsv_new(path_to_mtx_tsv_files_dir_patient2_test)


    # ### 1.5: **Remove samples from the matrix dataframe with no matching images in the image folder**
    # 
    # > Note: indices are being reset after this action, so a mapping of old to new column indices is returned: `column_mapping`.
    # 
    # > Note: the dataframe is also reordered according to the images order in the image folder

    matrix_dataframe_train, column_mapping_train = loadAndPreProcess.cut_samples_with_no_matching_image_and_reorder_df(matrix_df=matrix_dataframe_train, 
                                                                                                                    image_folder_of_the_df=imageFolder_train, 
                                                                                                                    barcodes_df=barcodes_dataframe_train)

    
    matrix_dataframe_test, column_mapping_test = loadAndPreProcess.cut_samples_with_no_matching_image_and_reorder_df(matrix_df=matrix_dataframe_test, 
                                                                                                                    image_folder_of_the_df=imageFolder_test, 
                                                                                                                    barcodes_df=barcodes_dataframe_test)


    # ### 1.6: **Remove less-informative genes**
    # 
    # > we define *less-informative* genes as genes with less than K counts over all samples
    # 
    # > `Base_value` is a parameter for the user's choice
    # 
    # > Note: indices are being reset after this action, so a mapping of old to new column indices is returned: `row_mapping`.

    # begin by asserting that our dataframes have the same genes to begin with using the metadata of features_dataframe
    assert features_dataframe_train['gene_names'].equals(features_dataframe_test['gene_names'])

    Base_value = 10
    matrix_dataframe_train, matrix_dataframe_test, row_mapping = loadAndPreProcess.cut_genes_with_under_B_counts_from_train_and_test(matrix_dataframe_train, matrix_dataframe_test, Base_value) 


    # ### 1.7: **Normalize matrix_dataframe entries**
    # 
    # > normaliztion will be performed on the remainning rows of the dataframe with the logic "log 1P"
    # 
    # > This method Calculates log(1 + x)

    matrix_dataframe_train = loadAndPreProcess.perform_log_1p_normalization(matrix_dataframe_train) 

    matrix_dataframe_test = loadAndPreProcess.perform_log_1p_normalization(matrix_dataframe_test) 


    # > We have performed all of the pre-processing actions on our matrix dataframes. (more pre-processing is still needed our datasets)
    # 
    # > print some information regarding our dataframes

    import projectUtilities
    projectUtilities.printInfoAboutReducedDF(matrix_dataframe_train)
    print("\n****\n")
    projectUtilities.printInfoAboutReducedDF(matrix_dataframe_test)


    # ### 1.8: **Create custom datasets**
    # 
    # > Each custom dataset is tailored per task
    # 
    # > there are four tasks: single gene prediction, k gene prediction, all gene prediction using NMF dim. reduction, all gene prediction using AE dim. reduction
    # 
    # > For each of the above tasks 2 datasets were created:
    # 
    # >> A Dataset created from the TRAIN data WITHOUT augmentation (without image transformations)
    # 
    # >> A Dataset created from the TRAIN data WITH augmentation (with image transformations)
    # 
    # >> A Dataset created from the TEST data WITHOUT augmentation (without image transformations)

    ## choose gene
    gene_name = 'CRISP3'  # was changed from 'BRCA1' because CRISP3 has the (almost) highest variance in both the train and test datasets.
                        # NOTE: the gene 'CRISP3' is "upregulated in certain types of prostate cancer" according to
                        #       https://www.genecards.org/cgi-bin/carddisp.pl?gene=CRISP3&keywords=rich
    from projectUtilities import get_variance_of_gene
    gene_variance_value = get_variance_of_gene(gene_name=gene_name, matrix_df=matrix_dataframe_train, row_mapping=row_mapping, features_df=features_dataframe_train)
    print(f'The chosen gene is {gene_name} and its variance is {gene_variance_value}')

    ## create datasets
    custom_DS_SingleValuePerImg_augmented = loadAndPreProcess.STDL_Dataset_SingleValuePerImg(imageFolder=augmentedImageFolder_train, 
                                                                matrix_dataframe=matrix_dataframe_train, 
                                                                features_dataframe=features_dataframe_train, 
                                                                barcodes_dataframe=barcodes_dataframe_train, 
                                                                column_mapping=column_mapping_train,
                                                                row_mapping=row_mapping,
                                                                chosen_gene_name=gene_name)
    custom_DS_SingleValuePerImg_test = loadAndPreProcess.STDL_Dataset_SingleValuePerImg(imageFolder=imageFolder_test, 
                                                                matrix_dataframe=matrix_dataframe_test, 
                                                                features_dataframe=features_dataframe_test, 
                                                                barcodes_dataframe=barcodes_dataframe_test, 
                                                                column_mapping=column_mapping_test,
                                                                row_mapping=row_mapping,
                                                                chosen_gene_name=gene_name)


    # <div class="alert alert-block alert-info">
    # <b>Note:</b> inside the init phase of `STDL_Dataset_KValuesPerImg_KGenesWithHighestVariance` class, K genes with the highest variance are chosen from matrix_dataframe, and they are the only genes that are kept for training and testing purposes
    # </div>

    k = 10

    custom_DS_KGenesWithHighestVariance_augmented = loadAndPreProcess.STDL_Dataset_KValuesPerImg_KGenesWithHighestVariance(imageFolder=augmentedImageFolder_train, 
                                                                            matrix_dataframe=matrix_dataframe_train, 
                                                                            features_dataframe=features_dataframe_train, 
                                                                            barcodes_dataframe=barcodes_dataframe_train, 
                                                                            column_mapping=column_mapping_train,
                                                                            row_mapping=row_mapping,
                                                                            num_of_dims_k=k)
    custom_DS_KGenesWithHighestVariance_test = loadAndPreProcess.STDL_Dataset_KValuesPerImg_KGenesWithHighestVariance(imageFolder=imageFolder_test, 
                                                                            matrix_dataframe=matrix_dataframe_test, 
                                                                            features_dataframe=features_dataframe_test, 
                                                                            barcodes_dataframe=barcodes_dataframe_test, 
                                                                            column_mapping=column_mapping_test,
                                                                            row_mapping=row_mapping,                                                                                                                  
                                                                            num_of_dims_k=k)

    # <div class="alert alert-block alert-info">
    # <b>Note:</b> inside the init phase of `STDL_Dataset_KValuesPerImg_LatentTensor_NMF` class, an NMF decompositionis performed on the matrix_dataframe object
    # </div>

    k = 10

    custom_DS_LatentTensor_NMF_augmented = loadAndPreProcess.STDL_Dataset_KValuesPerImg_LatentTensor_NMF(imageFolder=augmentedImageFolder_train, 
                                                                            matrix_dataframe=matrix_dataframe_train, 
                                                                            features_dataframe=features_dataframe_train, 
                                                                            barcodes_dataframe=barcodes_dataframe_train, 
                                                                            column_mapping=column_mapping_train,
                                                                            num_of_dims_k=k)
    custom_DS_LatentTensor_NMF_test = loadAndPreProcess.STDL_Dataset_KValuesPerImg_LatentTensor_NMF(imageFolder=imageFolder_test, 
                                                                            matrix_dataframe=matrix_dataframe_test, 
                                                                            features_dataframe=features_dataframe_test, 
                                                                            barcodes_dataframe=barcodes_dataframe_test, 
                                                                            column_mapping=column_mapping_test,
                                                                            num_of_dims_k=k)

    # <div class="alert alert-block alert-info">
    # <b>Note:</b> 
    # <ul>
    #   <li>first we create a dataset from `matrix_dataframe_train` to feed our AEnet.</li>
    #   <li>Then we create our AEnet and train it.</li>
    #   <li>Finally, we create our `custom_DS_LatentTensor_AE` class, in which the Autoencoder network will be saved.</li>
    # </ul>
    # </div>

    # TODO: uncomment later when AE tests are wanted
    dataset_from_matrix_df = loadAndPreProcess.STDL_Dataset_matrix_df_for_AE_init(matrix_dataframe_train)


    from executionModule import get_Trained_AEnet
    k = 10
    AEnet = get_Trained_AEnet(dataset_from_matrix_df=dataset_from_matrix_df, z_dim=k, num_of_epochs=30, device=device) #NOTE num of epochs - was raised to 40 because no convergence on 20

    k = 10
    custom_DS_LatentTensor_AE_augmented = loadAndPreProcess.STDL_Dataset_KValuesPerImg_LatentTensor_AutoEncoder(imageFolder=augmentedImageFolder_train, 
                                                                            matrix_dataframe=matrix_dataframe_train, 
                                                                            features_dataframe=features_dataframe_train, 
                                                                            barcodes_dataframe=barcodes_dataframe_train, 
                                                                            AEnet=AEnet,                                                                                                            
                                                                            column_mapping=column_mapping_train,
                                                                            num_of_dims_k=k,
                                                                            device=device)
    custom_DS_LatentTensor_AE_test = loadAndPreProcess.STDL_Dataset_KValuesPerImg_LatentTensor_AutoEncoder(imageFolder=imageFolder_test, 
                                                                            matrix_dataframe=matrix_dataframe_test, 
                                                                            features_dataframe=features_dataframe_test, 
                                                                            barcodes_dataframe=barcodes_dataframe_test, 
                                                                            AEnet=AEnet,                                                                                                       
                                                                            column_mapping=column_mapping_test,
                                                                            num_of_dims_k=k,
                                                                            device=device)


    # ### 1.9: prepare for the next phases in which the experiments are executed
    # 
    # > import `executionModule` which contains the experiments, training methods, and testing methods
    # 
    # > create `hyperparameters` dictionary which will contain all of the hyper-parameters for our experiments (note - user can change these later)
    # 
    # > create `model_list` that will hold all the names for the models that will be used (only 3 models for now, as can be seen below). the models are:
    # 
    # >> `BasicConvNet` model
    # 
    # >> `DensetNet121` model
    # 
    # >> `Inception_V3` model

    # <div class="alert alert-block alert-warning">
    # <b>Warning:</b> change the hyper-parameters below with caution if needed !
    # </div>

    


    import executionModule

    # define hyperparameters for the TRAINING of the models (NOT the testing phases of the experiments)
    hyperparameters = dict()
    hyperparameters['batch_size'] = 30
    hyperparameters['max_alowed_number_of_batches'] = 99999 #<--------------------------change to inf or 99999. anythin below 1220 will cut some batches ... this is only used to speed up training
    hyperparameters['precent_of_dataset_allocated_for_training'] = 0.8  # TODO currently not used
    hyperparameters['learning_rate'] = 1e-4
    hyperparameters['momentum'] = 0.9
    hyperparameters['num_of_epochs'] = 50 #<------------------------------------------change to 5 at least
    hyperparameters['num_workers'] = 2 #<------------------------------------------ NOTE: default is 0, means everything happens serially. testing 2 now !
                                       # see: https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading

    # define hyperparameters for BsicConvNet
    hyperparameters['channels'] = [32, 32, 64, 64] 
    hyperparameters['num_of_convolution_layers'] = len(hyperparameters['channels'])
    hyperparameters['hidden_dims'] = [100, 100]
    hyperparameters['num_of_hidden_layers'] = len(hyperparameters['hidden_dims'])
    hyperparameters['pool_every'] = 99999

    # add the chosen single gene's index to the hyperparameters
    from projectUtilities import get_index_of_gene_by_name
    hyperparameters['gene_name'] = gene_name
    hyperparameters['geneRowIndexIn_Reduced_Train_matrix_df'], _ = get_index_of_gene_by_name(gene_name=gene_name, matrix_df=matrix_dataframe_train, row_mapping=row_mapping, features_df=features_dataframe_train)
    hyperparameters['geneRowIndexIn_Reduced_Test_matrix_df'], _ = get_index_of_gene_by_name(gene_name=gene_name, matrix_df=matrix_dataframe_test, row_mapping=row_mapping, features_df=features_dataframe_test)

    # list of all models used
    model_list = []
    model_list.append('BasicConvNet')
    model_list.append('DensetNet121')


    # <div class="alert alert-block alert-info">
    # <b>Note:</b> In each experiment, the model is trained with the augmented train dataset, and then tested on the test dataset
    #     (NMF and AE experiments also test on the regular train dataset after training is done)
    # </div>

    # ## Phase 2: Single Gene Prediction

    # TODO: commented because already executed in batch
#    hyperparameters['num_of_epochs'] = 20
#
#    executionModule.runExperiment(ds_train=custom_DS_SingleValuePerImg_augmented, 
#                                ds_test=custom_DS_SingleValuePerImg_test,
#                                hyperparams=hyperparameters, 
#                                device=device, 
#                                model_name='BasicConvNet', 
#                                dataset_name='single_gene')
#
#    hyperparameters['num_of_epochs'] = 40
#
#    executionModule.runExperiment(ds_train=custom_DS_SingleValuePerImg_augmented, 
#                                ds_test=custom_DS_SingleValuePerImg_test,
#                                hyperparams=hyperparameters, 
#                                device=device, 
#                                model_name='DensetNet121', 
#                                dataset_name='single_gene')



#    # ## Phase 3: K genes prediction
#
#    hyperparameters['num_of_epochs'] = 20
#
#    executionModule.runExperiment(ds_train=custom_DS_KGenesWithHighestVariance_augmented, 
#                                ds_test=custom_DS_KGenesWithHighestVariance_test,
#                                hyperparams=hyperparameters, 
#                                device=device, 
#                                model_name='BasicConvNet', 
#                                dataset_name='k_genes')
#
#
#    hyperparameters['num_of_epochs'] = 40
#
#    executionModule.runExperiment(ds_train=custom_DS_KGenesWithHighestVariance_augmented, 
#                                ds_test=custom_DS_KGenesWithHighestVariance_test,
#                                hyperparams=hyperparameters, 
#                                device=device, 
#                                model_name='DensetNet121', 
#                                dataset_name='k_genes')
#
#
#    # ## Phase 4: All genes prediction - using dimensionality reduction techniques
#    # 
#    # ### 4.1: Prediction using dimensionality reduction technique NMF
#
#    hyperparameters['num_of_epochs'] = 20
#
#    executionModule.runExperiment(ds_train=custom_DS_LatentTensor_NMF_augmented, 
#                                ds_test=custom_DS_LatentTensor_NMF_test,
#                                hyperparams=hyperparameters, 
#                                device=device, 
#                                model_name='BasicConvNet', 
#                                dataset_name='NMF')
#
#
#    hyperparameters['num_of_epochs'] = 40
#
#    executionModule.runExperiment(ds_train=custom_DS_LatentTensor_NMF_augmented, 
#                                ds_test=custom_DS_LatentTensor_NMF_test,
#                                hyperparams=hyperparameters, 
#                                device=device, 
#                                model_name='DensetNet121', 
#                                dataset_name='NMF')


    # ### 4.2: Prediction using dimensionality reduction technique AE

    
    hyperparameters['num_workers'] = 0     # !!!
    hyperparameters['num_of_epochs'] = 20

    executionModule.runExperiment(ds_train=custom_DS_LatentTensor_AE_augmented, 
                                ds_test=custom_DS_LatentTensor_AE_test,
                                hyperparams=hyperparameters, 
                                device=device, 
                                model_name='BasicConvNet', 
                                dataset_name='AE')

    hyperparameters['num_of_epochs'] = 40

    executionModule.runExperiment(ds_train=custom_DS_LatentTensor_AE_augmented, 
                                ds_test=custom_DS_LatentTensor_AE_test,
                                hyperparams=hyperparameters, 
                                device=device, 
                                model_name='DensetNet121', 
                                dataset_name='AE')


    # <div class="alert alert-block alert-danger">
    # <b>Note:</b> below this - everything is a testing block
    # </div>



if __name__=="__main__":
    main()
