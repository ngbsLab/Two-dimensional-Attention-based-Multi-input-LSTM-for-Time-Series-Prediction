# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:37:25 2018

@author: Xingguang Zhang
"""
from keras.layers import LSTM,Dense,Activation,Softmax,Lambda,RepeatVector,\
            Add,Reshape,Multiply,Concatenate,Dot
import keras.backend as K
from keras.constraints import maxnorm
from keras import regularizers

def first_attention(In1, h0, s0, m, T, numFeature):
    X_inv = Lambda(lambda x: K.permute_dimensions(In1, (0,2,1)))(In1)  
    UedotXk = Dense(T, use_bias = False)(X_inv)
    for t in range(T):
        x = Lambda(lambda x: In1[:,t,:])(In1)  
        if t == 0:
            x_lstm = Reshape((1,numFeature))(x)
            h, _, s = LSTM(m, return_state = True)(x_lstm, initial_state = [h0, s0])
            H = Reshape((1, m))(h)
        else:
            x_lstm = Reshape((1,numFeature))(x_lstm)
            h, _, s = LSTM(m, kernel_constraint=maxnorm(4.),kernel_regularizer = regularizers.l2(0.01),\
                           return_state = True)(x_lstm, initial_state = [h, s])
            ht = Reshape((1, m))(h)
            H = Concatenate(axis = 1)([H, ht])
        conc = Concatenate()([h, s])
        WedotHs = Dense(T, use_bias = False)(conc)
        WedotHs = RepeatVector(numFeature)(WedotHs)
        add = Add()([WedotHs, UedotXk])
        acti = Activation(activation = 'tanh')(add)
        e = Dense(1, use_bias = False)(acti)
        e = Reshape((numFeature,))(e)
        alpha = Softmax(axis = 1)(e)
        x_lstm = Multiply()([x, alpha])
    return H

def second_attention(H, Iny, hd0, sd0, m, p, T):
    UddotXk = Dense(m, use_bias = False)(H)
    d = hd0
    c = sd0
    for t in range(T-1):
        y = Lambda(lambda x: Iny[:,t])(Iny) 
        y = Reshape((1,))(y)
        conc = Concatenate()([d, c]) 
        WddotHs = Dense(m, use_bias = False)(conc)
        WddotHs = RepeatVector(T)(WddotHs)
        add = Add()([WddotHs, UddotXk])
        acti = Activation(activation = 'tanh')(add)
        l = Dense(1, use_bias = False)(acti)
        l = Reshape((T,))(l)
        beta = Softmax(axis = 1)(l)
        context = Dot(axes = 1)([beta,H])                                       #(None,m)
        cy = Concatenate()([y, context])
        new_y = Dense(1, use_bias = True)(cy)
        new_y = Reshape((1, 1))(new_y)
        d, _, c = LSTM(p, kernel_constraint=maxnorm(4.),kernel_regularizer = regularizers.l2(0.01),\
                       return_state = True)(new_y, initial_state = [d, c])
    dc = Concatenate()([d, c]) 
    Wydotdc = Dense(p, use_bias = True)(dc)
    Y = Dense(1, use_bias = True)(Wydotdc)
    return Y