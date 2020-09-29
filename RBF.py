#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/9/29 3:59
@author: merci
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(7) # cpu
torch.cuda.manual_seed(7) #gpu
np.random.seed(7) #numpy

class RBF(nn.Module):

    def __init__(self, indim ,centers, outdim):
        super(RBF, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.centers = centers

        self.beta = 8
        self.l2_norm = nn.PairwiseDistance(p=2)

    def maxDis(self, centers):
        maxdis = 0
        for i in centers:
            distance = self.l2_norm(i,centers)
            if distance.max()>maxdis:
                maxdis = distance.max()
        return maxdis


    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # calculate activations of RBFs
        s = (X.size(0),16,33)
        d_max = self.maxDis(self.centers)
        sigma = d_max / np.sqrt(2 * self.centers.size(0))
        x = X.unsqueeze(1).expand(s)
        c = self.centers.unsqueeze(0).expand(s)
        distance_2 = (x - c).pow(2).sum(-1)
        bs = torch.ones_like(Y).cuda()
        G = torch.exp(-distance_2 / (2 * sigma ** 2))
        G = torch.cat((G,bs),1)

        # calculate output weights (pseudoinverse)
        self.W = torch.matmul(torch.matmul(torch.inverse(torch.matmul(G.T, G)), G.T),Y.double())
        return self.W

    def test(self, X):
        """ X: matrix of dimensions n x indim """
        s = (X.size(0),16,33)
        d_max = self.maxDis(self.centers)
        sigma = d_max / np.sqrt(2 * self.centers.size(0))
        x = X.unsqueeze(1).expand(s)
        c = self.centers.unsqueeze(0).expand(s)
        bs = torch.ones((X.size(0),1)).cuda()
        distance_2 = (x - c).pow(2).sum(-1)
        G = torch.exp(-distance_2 / (2 * sigma ** 2))
        G = torch.cat((G, bs), 1)
        Y = torch.matmul(G,self.W)
        predictions = torch.tensor([1 if i>0 else -1 for i in Y])
        return predictions