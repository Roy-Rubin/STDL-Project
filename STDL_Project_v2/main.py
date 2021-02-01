
from STDLclass import STDLObject


def main():
    print("Hello World !\nGot to main file. Starting to work !")
    
    stdl = STDLObject(path_to_all_data_folders_dir='/home/roy.rubin/STDLproject/spatialGeneExpressionData/', 
                            chosen_gene_name='ENSG00000096006')
    
    stdl.getModelTrainingStatus()

    stdl.trainModel(name_of_model_to_train='BasicConvNet')          # add `hyperparameters` if wanted
    stdl.runModelOnTestData(name_of_model_to_run='BasicConvNet') 

    stdl.getModelTrainingStatus()
    
    stdl.trainModel(name_of_model_to_train='DenseNet121')           # add `hyperparameters` if wanted
    stdl.runModelOnTestData(name_of_model_to_run='DenseNet121')

    stdl.getModelTrainingStatus()
    

if __name__=="__main__":
    main() 