# STDL-Project

> Spatial Transcriptomics (ST) Deep Learning Project

> Prediction of gene expression levels through biopsy image analysis combined with ST data


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
- Experiments performed:
    - Given a fraction of a biopsy image - Perform a prediction of a single gene's expression level 
    - Given a fraction of a biopsy image - Perform a prediction of K chosen genes expression levels
    - Given a fraction of a biopsy image - Perform a prediction of all of the genes expression levels
        - Performed with dimensionality reduction technique: `non-negative matrix factorization (NMF)`
        - Performed with dimensionality reduction technique: `autoencoder deep neural network (AE)`

## Contents-of-the-repository

> File: `loadAndPreProcess.py`. contains ...

> File: `executionModule.py`. contains ...

> File: `deepNetworkArchitechture.py`. contains ...

> File: `projectUtilities.py`. contains ...

> File: `STDL_notebook2.ipynb`. contains ...

> anythin else - needs to be deleted !!!!! TODO !!!!!!!!!!!!


## Requirements

> note that this is what i had on my computation node while performing the experiments

- python
- pytorch
- sklearn
- pandas=1.0.4
- numpy=1.18.5
- torchvision
Python 3.7.6
numpy==1.18.5
pandas==1.0.4
scikit-image==0.17.2
scikit-learn==0.23.1
scipy==1.4.1
torch = 1.3.0
torchvision == 0.4.1a0+d94043a
Pillow==7.2.0
matplotlib==3.2.1
opencv-python (cv2)==4.4.0.42

## How-to-use

### Step 1

- **Install all required libraries**
    - do that

### Step 2

- **Run notebook**
    - better performance when ran on a gpu

### Step 3

- **Enjoy**
    - better performance when ran on a gpu

- **Consider using different hyperparameter values**
    - better performance when ran on a gpu

## FAQ

- **How can I get the data ?**
    - good question :smile: ?
- **This or that doesnt work... what to do?**
    - No problem! Just do this.
- b

## Thanks

- Project supervisor: Leon Anavy
- Computational power: GPUs used are from CS faculty @ Technion - Israeli institute of technology
- Original idea from the paper below:
> [**Integrating spatial gene expression and breast tumour morphology via deep learning**](https://rdcu.be/b46sX)<br/>
  by Bryan He, Ludvig Bergenstråhle, Linnea Stenbeck, Abubakar Abid, Alma Andersson, Åke Borg, Jonas Maaskola, Joakim Lundeberg & James Zou.<br/>
  <i>Nature Biomedical Engineering</i> (2020).

> see their code @ thier girhub repository <a href="https://github.com/bryanhe/ST-Net" target="_blank">`link here`</a>

- Credit to <a href="https://gist.githubusercontent.com/fvcproductions/1bfc2d4aecb01a834b46/raw/370c1944e2767e620fca720e8ee51042652727cd/sampleREADME.md" target="_blank">`this person ... `</a> for formatting tips on the readme
- Thanks in advance to anyone I forgot...

## License

- Data - 10x genomics (tag ?) (Credit ?)
- Technion ?
