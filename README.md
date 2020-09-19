# STDL-Project

> Spatial Transcriptomics (ST) Deep Learning Project

> Prediction of gene expression levels through biopsy image analysis combined with ST data

## Table of Contents
- [Abstract](#Abstract)
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

## Requirements

- python
- pytorch
- sklearn
- pandas
- numpy

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
- Credit to <a href="https://gist.githubusercontent.com/fvcproductions/1bfc2d4aecb01a834b46/raw/370c1944e2767e620fca720e8ee51042652727cd/sampleREADME.md" target="_blank">`this person ... `</a> for formatting tips on the readme
- Thanks in advance to anyone I forgot...

## License

- Data - 10x genomics (tag ?) (Credit ?)
- Technion ?
