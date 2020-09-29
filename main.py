#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/9/29 0:26
@author: merci
"""
import numpy as np
import torch
from SOM import SOM
from RBF import RBF
import scipy.io as scio
from sklearn import svm


torch.manual_seed(7) # cpu
torch.cuda.manual_seed(7) #gpu
np.random.seed(7) #numpy
def SOM_RBF():
    train = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_train = scio.loadmat('data_train.mat')
    data_test = scio.loadmat('data_test.mat')
    data_label = scio.loadmat('label_train.mat')
    X = torch.tensor(data_train['data_train']).to(device)
    train_label = torch.tensor(data_label['label_train']).to(device)
    test_data = torch.tensor((data_test['data_test'])).to(device)
    print('Building SOM model...')
    model = SOM(input_size=33, hidden_size=16, sigma=np.sqrt(81))
    model = model.to(device)
    if train == True:
            losses = list()
            for epoch in range(1000):
                running_loss = 0

                loss = model.self_organizing(X, epoch)    # so phase
                running_loss += loss

                losses.append(running_loss)
                print('epoch = %d, loss = %.2f' % (epoch + 1, running_loss))
            for epoch in range(30000):
                running_loss = 0

                loss = model.convergence(X)    # conv phase
                running_loss += loss

                losses.append(running_loss)
                print('epoch = %d, loss = %.2f' % (epoch + 1, running_loss))
            train = False
    feedback = model.plot_point_map(X,train_label,['Class 0', 'Class 1'])
    a,b,c = feedback
    tb = (a-b)/a
    tc = (a-c)/a
    accuracy0 = np.maximum(tb,tc).mean()
    center_vectors= model.weight.T
    print('Building RBF model...')
    RBF_model = RBF(33, center_vectors, 1)
    RBF_model = RBF_model.to(device)
    weight = RBF_model.train(X,train_label)
    predictions = RBF_model.test(X).unsqueeze(1)
    accuracy1 = (predictions==train_label.cpu()).sum()/torch.tensor(X.size(0)).float()
    predictions_y = RBF_model.test(test_data)
    print(accuracy1)
    return accuracy0,accuracy1,predictions_y

def SVM():
    data_train = scio.loadmat('data_train.mat')
    data_test = scio.loadmat('data_test.mat')
    data_label = scio.loadmat('label_train.mat')
    X = data_train['data_train']
    y = np.squeeze(data_label['label_train'])
    test = data_test['data_test']
    print('Building SVM model')
    clf = svm.SVC()
    clf.fit(X,y)
    predictions = clf.predict(X)
    accuracy = (predictions==y).sum().__float__() / X.shape[0]
    predictions_y = clf.predict(test)
    print(accuracy)
    return accuracy, predictions_y

if __name__ == '__main__':
    mode = 0
    if mode == 0:
        SOM_RBF()
    else:
        SVM()