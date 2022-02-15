# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:24:14 2018

@author: Xingguang Zhang
"""
import numpy as np
import pandas as pd

def create_dataset(dataset, numStepTrain, window_size, label_column):
    trainX, trainY, testX, testY = [], [], [], []
    numfeatures = np.size(dataset,1)
    if numStepTrain < 1:
        numStepTrain = int(len(dataset) * numStepTrain)
    #split out train set
    dataTrain = dataset[0:numStepTrain]
    #normalization
    sig, mu, dataTrainStd = normalize(dataTrain)
    #create train and test dataset
    for i in range(numStepTrain - window_size):
        a = dataTrainStd[i:(i+window_size), :]
        trainX.append(a)
        trainY.append(dataTrainStd[(i + 1) : (i + window_size + 1), label_column])
    for j in range((numStepTrain- window_size),(len(dataset)-window_size)):
        a = dataset[j : (j + window_size),:]
        testX.append(a)
        testY.append(dataset[(j + 1) : (j + window_size + 1), label_column])
    #reshape the dataset to the standard shape of LSTM input
    trainX, trainY, testX, testY = np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], numfeatures))
    testX = (testX - mu) / sig
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], numfeatures))
    testY = (testY - mu) / sig
    return sig, mu, trainX, trainY, testX, testY

def normalize(train_data):
    sig = np.std(train_data)
    mu = np.mean(train_data)
    dataTrainStd = (train_data - mu) / sig
    return sig, mu, dataTrainStd

def read_data(path, steps, no_column):
    whole = np.array(pd.read_csv(path))
    data = whole[::steps,no_column]
    data = np.reshape(data, [np.size(data,0),np.size(no_column)])
    for k in range(data.shape[1]):
        # for the unrecorded data(nan), padding it with that of the previous time step 
        for i in range(data.shape[0]):
            if np.isnan(data[i,k]):
                data[i,k] = data[i-1,k]
    return data.astype('float32')
