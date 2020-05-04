#!/usr/bin/python3
# -*- coding:utf-8 -*-
# author: zhaofeng-shu33
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from utility import int2bin
import os
import pdb
WEIGHT_FILE_NAME = 'weight_838.h5'
IGNORE_FIT = False
if __name__ == '__main__':
    m = Sequential()
    m.add(Dense(3, input_dim = 8, activation='tanh'))
    m.add(Dense(8, activation='sigmoid'))
    X = np.eye(8)
    m.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
    pdb.set_trace()
    if(os.path.exists(WEIGHT_FILE_NAME)):
        m.load_weights(WEIGHT_FILE_NAME)
        pre_eval = m.evaluate(X,X)
        if(abs(pre_eval[1]-1)<0.01):
            IGNORE_FIT = True
    if not(IGNORE_FIT):
        print('training...')
        m.fit(X, X, batch_size=8, epochs=10000, verbose=0)
        m.save_weights(WEIGHT_FILE_NAME)
    print('evaluating...')
    print(m.evaluate(X,X))
    # construct evaluation dataset
    X_eval = np.zeros([256, 8])
    for i in range(256):
        X_eval[i,:] = int2bin(i)
    Y = m.predict(X_eval) > 0.5
    X_eval = X_eval.astype(np.bool)
    acc_cnt = np.zeros(9)
    for i in range(256):
        error_bit = int(8 - np.sum(X_eval[i,:]==Y[i,:]))
        acc_cnt[error_bit] += 1
    print('evaluation result',acc_cnt)
