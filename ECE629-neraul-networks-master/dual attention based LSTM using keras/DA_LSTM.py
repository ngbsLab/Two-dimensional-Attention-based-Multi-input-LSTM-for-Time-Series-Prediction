# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:27:27 2018

@author: Xingguang Zhang
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.layers import Input
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras import optimizers
#from keras import regularizers
from data_creation import create_dataset, read_data
from dual_attention import first_attention, second_attention

def scheduler(epoch):
    # every 20 epochs, lr reduced to 20%
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.2)
        print("lr changed to {}".format(lr * 0.2))
    return K.get_value(model.optimizer.lr)

np.random.seed(8)
path = 'full_non_padding.csv'
#sample the data for every 20 minutes to reduce the amount of data
sample_step = 5
#apple, adobe, amazon,microsoft,netflix
in_column = (1,2,11,62,67)
out_column = 67

for i in range(np.size(in_column)):
    in_column = list(in_column)
    label_column = 0
    if in_column[i] == out_column:
        label_column = i
        break
        
data = read_data(path, sample_step, in_column)
data_attention = np.reshape(data[:,label_column] ,(-1, 1))
#create dataset
numFeature = np.size(data,1) #n
#Time steps (window size)
T = 20
#cells number in encoder
m = 64
#cells number in decoder
p = 64
# prepare inputs
sig, mu, trainX, trainY, testX, testY = create_dataset(data,\
                        numStepTrain = 0.7, window_size = T, label_column = label_column)

train_label = np.reshape(trainY[:, -1], (-1, 1))
test_label = np.reshape(testY[:, -1], (-1, 1))
h_init = np.zeros((trainX.shape[0], m))
s_init = np.zeros((trainX.shape[0], m))
hd_init = np.zeros((trainX.shape[0], p))
sd_init = np.zeros((trainX.shape[0], p))

# define input variables
In1 = Input(shape = (trainX.shape[1], trainX.shape[2]))         #(None, T, m)
Iny = Input(shape = (trainY.shape[1],))                         #(None, T)
h0 = Input(shape = (m,))                                        #(None, m)
s0 = Input(shape = (m,))
hd0 = Input(shape = (p,))                                       #(None, m)
sd0 = Input(shape = (p,))

#attention 1st stage --- encoder
H = first_attention(In1, h0, s0, m, T, numFeature)              #(None, T, m)
#attention 2nd stage
Y = second_attention(H, Iny, hd0, sd0, m, p, T)              

reduce_lr = LearningRateScheduler(scheduler)
#build model
model = Model(inputs = [In1, Iny, h0, s0, hd0, sd0], outputs = Y)
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer = optimizer)
#model.summary()
model.fit([trainX, trainY, h_init, s_init, hd_init, sd_init], train_label, epochs=120, batch_size=128, \
          callbacks=[reduce_lr], verbose=1, shuffle=False)

trainPredict = model.predict([trainX, trainY, h_init, s_init, hd_init, sd_init])
testPredict = model.predict([testX, testY, h_init, s_init, hd_init, sd_init])

trainPredict = trainPredict * sig + mu
train_label = train_label * sig + mu
testPredict = testPredict * sig + mu
test_label = test_label * sig + mu

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_label, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_label, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(data_attention)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[T : len(trainPredict) + T, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data_attention)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + T : len(data_attention), :] = testPredict
# plot baseline and predictions
plt.figure(1,figsize=(12,9))
plt.plot(data_attention, label = 'ground truth')
plt.plot(trainPredictPlot, label = 'train prediction')
plt.plot(testPredictPlot, label = 'test prediction')
plt.legend()
plt.show()
plt.figure(1, figsize=(12, 9))
plt.plot(test_label, label = 'test target')
plt.plot(testPredict, label = 'prediction')
plt.legend()
plt.show()
