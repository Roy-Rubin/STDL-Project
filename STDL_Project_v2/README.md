# STDL-Project (V2)

> Spatial Transcriptomics Deep Learning (STDL) Project

> Prediction of gene expression levels through biopsy image analysis combined with Spatial Transcriptomics (ST) data


## Table of Contents
- [Abstract](#Abstract)
- [Contents-of-the-repository](#Contents-of-the-repository)
- [Requirements](#Requirements)
- [How-to-use](#How-to-use)
- [FAQ](#FAQ)
- [Thanks](#Thanks)
- [License](#License)

## Abstract

- Spatial transcriptomics is a recent technique used to capture the spatial distribution of messenger RNA sequences within tissue sections.
- The technique is used on a biopsy sample to produce a matrix of gene expression levels for that sample.
- Deep learning computer vision algorithms are introduced to analyze the biopsy image.
- When combined with the gene expression levels matrix, a trained model is produced that learns the connection between a matrix entry and its relevant image.
- The goal is, to be able to produce a close-as-posibble approximation to real gene expression levels, using the biopsy image alone.
- Given a fraction of a biopsy image - Perform a prediction of a single gene's expression level 

> `NOTE`: this is the second version of the project. in this version:
>> only a single gene is predicted.
>> a new wrapping class helps organize functionality: `STDLObject`
>> some file names have changed


## Contents-of-the-repository

> File: `STDLclass.py`. contains the new `STDLObject` class that assists in organizing every functionality necessary for training and testing

> File: `projectLoadAndPreProcess.py`. contains all functions to load data, create custom datasets, and perform pre processing actions

> File: `projectTrainAndPredict.py`. contains all functions required for training, testing, predicting, and performing experiments

> File: `projectModels.py`. contains a few basic deep learning architechtures used in the project. others more complex architectures will be imported from `torchvision`

> File: `projectUtilities.py`. contains assisting functions for information printing and experimentation

> File: `environment.yml`. contains the conda instalation instructions for the project

> File: `main.py`. contains a usage example for the STDL class object


## Requirements

> All of the requirements can be seen in the `environment.yml` file


## How-to-use

### Step 1

- **Install all required libraries**
    - Preferably use conda to install the environment
    - Use conda code below to create the environemnt and install all of the required packages
    ```shell
    $ conda env create -f environment.yml
    ```

### Step 2

- **Activate the environment**
    - The environment is called "STDL"
    - Use the code below to activate it:
     ```shell
    $ conda activate STDL
    ```

- **Run notebook**
    - preferably with jupyter lab

### Step 3

- **Enjoy**
    - better performance when ran on a gpu

- **Consider using different hyperparameter values**
    - You can try for example to increase the amount of trainning epochs performed.

## FAQ

- **How can I get the data ?**
    - Please contact <a href="https://10xgenomics.com/" target="_blank">`10xgenomics data`</a>. Part of the data was acquired from their website.
    - Please see <a href="https://data.mendeley.com/datasets/29ntw7sh4r/2" target="_blank">`Mendeley data`</a>. Part of the data was acquired from their website.
- **This or that doesnt work... what to do?**
    - No problem! Contact me for questions.

## Thanks

- Project supervisor: Leon Anavy
- Computational power: GPUs used are from CS faculty @ Technion - Israeli institute of technology
- Original idea from the paper below:
> [**Integrating spatial gene expression and breast tumour morphology via deep learning**](https://rdcu.be/b46sX)<br/>
  by Bryan He, Ludvig Bergenstråhle, Linnea Stenbeck, Abubakar Abid, Alma Andersson, Åke Borg, Jonas Maaskola, Joakim Lundeberg & James Zou.<br/>
  <i>Nature Biomedical Engineering</i> (2020).

>> See their code @ their github repository <a href="https://github.com/bryanhe/ST-Net" target="_blank">`link here`</a>

- Credit to <a href="https://gist.githubusercontent.com/fvcproductions/1bfc2d4aecb01a834b46/raw/370c1944e2767e620fca720e8ee51042652727cd/sampleREADME.md" target="_blank">`this person`</a> for formatting tips on the readme
- Thanks in advance to anyone I forgot...

## License

- Data acquired from <a href="https://10xgenomics.com/" target="_blank">`10xgenomics`</a> and <a href="https://data.mendeley.com/datasets/29ntw7sh4r/2" target="_blank">`Mendeley`</a>
- Project performed @ Technion - Israeli institute of technology
