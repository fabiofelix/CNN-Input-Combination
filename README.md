# Enhancing Sound-Based Classification of Birds and Anurans with Spectrogram Representations and Acoustic Indices in Neural Network Architectures

This repo contains a code implementation of the input combinations proposed by a [paper](https://) submitted to [Ecological Informatics](https://www.sciencedirect.com/journal/ecological-informatics). It combines spectrogram representations and other features to train models to classify [natural sounds](https://doi.org/10.1007/s10980-011-9600-8) of birds and anurans species.

## Prerequisites

[![Python 3.8.10](https://img.shields.io/badge/python-3.8.10-green.svg)](https://www.python.org/downloads/release/python-3810/) [![tensorflow 2.7.0](https://img.shields.io/badge/tensorflow-2.7.0-orange.svg)](https://pypi.org/project/tensorflow/2.7.0/)  [![Keras 2.7.0](https://img.shields.io/badge/keras-2.7.0-red.svg)](https://pypi.org/project/keras/2.7.0/) 
[![pandas 1.2.0](https://img.shields.io/badge/pandas-1.2.0-purple.svg)](https://pypi.org/project/pandas/1.2.0/) [![scikit-learn 1.0.1](https://img.shields.io/badge/scikit_learn-1.0.1-blue.svg)](https://pypi.org/project/scikit-learn/1.0.1/)

## Description

We created four Python files: *main.py* has the main functions we used to train and evaluate our models; *models.py* has the architectures and their variations; *quantification.py* has the implementation of a [custom loss function](https://doi.org/10.1007/s13748-016-0103-3); and *utils.py* has some auxiliary functions used to train/apply the models. We put a simple generator function, but you should use your generators.

The main code has the following parameters:

```python
  -a  Action to be performed, to train a model (train) or to appply some pretrained model (apply) 
  -s  Source directory to load spectrogram images.
  -t  Target directory to save/load model. Default = current directory
  -m  Model index 
      (0) SimpleCNN, 
      (1) BirdVox, 
      (2) ResNet, 
      (3) Inception
  -l  Quantity of labels that the model needs to classify
  -e  Quantity of epochs. Default = 100
  -b  Batch size used for training, validation and test. Default = 80
  -quant Add quantification loss function. Detault = None 
          (None) uses default categorical cross entropy
          (1) quantificaton with first weighting, 
          (2) quantificaton with second weighting, 
          (3) quantificaton with third weighting case
  -comb Especifies the type of input combination. Default = 0
        (0) None,
        (1) Batch normalization layer,
        (2) Dense layer,
        (3) Batch normalization, dense layers,
        (10) Three inputs,
        (11) Three inputs + Batch normalization layer,
        (12) Three inputs + Dense layer,
        (13) Three inputs + Batch normalization, dense layers,      
        (20) Three channels,
        (21) Three channels + Batch normalization layer,
        (22) Three channels + Dense layer,
        (23) Three channels + Batch normalization, dense layers    
  -hc Path of a file with the handcrafted features
  -aux Directory list to load three inputs/channels images. Concatenated with [-s]  Source directory
  -eval Generate model evaluation. Default = False.
```

## Running

To **train** with your dataset, the generator looks for two folders inside the source path: train and validation. Inside these folders, put your spectrogram images and a CSV file with two columns: file and label related to your spectrograms. Data folder has examples of this files/structure.

To **apply** our pre-trained models to your data, set target *-t* parameter with a folder that contains a model (inside *models* folder exists a link to Google Drive with our models) and set source *-s* parameter with your spectrograms folder. Code will generate a CSV with the predicted labels in the current path.

When training and applying the models to different inputs, you should inform *-aux* parameter (spectrograms) and/or *-hc* parameter (path to a CSV file with handcrafted features). Check the examples.

Handcrafted features should be in a CSV file with *file* (equal to the name of the image files) and *label* columns, and columns with the feature values.

Besides, you can use a ground truth to evaluate the models. Put a CSV file with two columns (file and expected labels) inside the source path and pass *-eval* parametter.

## Examples

```python
  python main.py -a train -l 15 -m 0 -e 100 -b 80 -quant -comb 0 -s /home/user/Desktop/data/spec -t /home/user/Desktop/model
```  

```python
  python main.py -a apply -l 15 -m 1 -e 100 -b 80 -quant -comb 3 -s /home/user/Desktop/data/mel -hc /home/user/Destop/data/features/features_test.csv -t /home/user/Desktop/model 
```  

```python
  python main.py -a train -l 15 -m 2 -e 100 -b 80 -quant -comb 10 -s /home/user/Desktop/data/ -aux spec mel pcen -t /home/user/Desktop/model 
```

```python
  python main.py -a apply -l 15 -m 3 -e 100 -b 80 -quant -comb 23 -s /home/user/Desktop/data/ -aux spec mel pcen -hc /home/user/Destop/data/features/features_test.csv -t /home/user/Desktop/model 
```

## Data used

We ran experiments with two datasets.
The **main dataset** is a subset of a database collected by the [LEEC lab](https://github.com/LEEClab) and other subsets were used in other papers, such as [[1]](https://doi.org/10.1016/j.ecolind.2020.107050), [[2]](https://doi.org/10.1016/j.ecolind.2020.107316), [[3]](https://doi.org/10.3390/info12070265), and [[4]](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2023.1283719/full). 
Our subset is labeled with animal species and will be available on the [lab website](https://github.com/LEEClab) as soon as possible.
The **additional dataset** is a subset of the [DCASE 2024](https://dcase.community/challenge2024/task-few-shot-bioacoustic-event-detection).
We also used samples from the [Google Audioset](http://research.google.com/audioset/download.html)

## Contact

* [Fábio F. Dias](https://scholar.google.com.br/citations?hl=pt-BR&user=uQ_qg2MAAAAJ) - e-mail: <ffd2011@nyu.edu>

* [Moacir A. Ponti](https://scholar.google.com.br/citations?user=ZxQDyNcAAAAJ&hl=pt-BR&oi=sra) - e-mail: <moacir@icmc.usp.br>

* [Rosane Minghim](https://scholar.google.com.br/citations?user=TodwpSwAAAAJ&hl=pt-BR&oi=ao) - e-mail: <rosane.minghim@ucc.ie>  
