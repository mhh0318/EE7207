#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/9/28 13:42
@author: merci
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.patches as mptchs
import matplotlib.pyplot as plt

torch.manual_seed(7) # cpu
torch.cuda.manual_seed(7) #gpu
np.random.seed(7) #numpy

class SOM(nn.Module):
    def __init__(self, input_size, hidden_size, lr=0.1, sigma=None):
        super(SOM, self).__init__()
        self.lr = lr
        self.sigma = sigma
        self.inputsize = input_size
        self.outsize = hidden_size

        self.weight = nn.Parameter(torch.randn(input_size, hidden_size), requires_grad=False)
        self.locations = nn.Parameter(torch.Tensor(list(self.paired_index())), requires_grad=False)
        self.l2_norm = nn.PairwiseDistance(p=2)

    def paired_index(self):
        for x in range(int(pow(self.outsize,0.5))):
            for y in range(int(pow(self.outsize,0.5))):
                yield (x, y)
    def _neighborhood_fn(self, d_ij, sigma):
        '''e^(-(input / sigma^2))'''
        d_ij.div_(2* sigma ** 2)
        d_ij.neg_()
        d_ij.exp_()
        return d_ij

    def forward(self, input):
        data_size = input.size()[0]
        input = input.view(data_size, -1, 1)
        batch_weight = self.weight.expand(data_size, -1, -1)

        dists = self.l2_norm(input, batch_weight)
        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)
        bmu_locations = self.locations[bmu_indexes]

        return bmu_locations, losses.sum().div_(data_size).item()

    def self_organizing(self, input, current_iter):

        batch_size = input.size()[0]
        #Set learning rate
        init_lr = self.lr
        lr = init_lr*np.exp(-current_iter/1000)

        #Set neighborhood func
        init_sigma = self.sigma
        t_cons = 1000/np.log(init_sigma)
        sigma = init_sigma*np.exp(-current_iter/t_cons)


        #Find best matching unit
        bmu_locations, loss = self.forward(input)

        distance_squares = self.locations.float() - bmu_locations.float()
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)

        lr_locations = self._neighborhood_fn(distance_squares, sigma)
        lr_locations.mul_(lr).unsqueeze_(1)

        delta = lr_locations * (input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0)
        delta.div_(batch_size)
        self.weight.data.add_(delta)

        return loss

    def convergence(self, input):
        lr = 1e-2
        bmu_locations, loss = self.forward(input)
        distance_squares = self.locations.float() - bmu_locations.float()
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)
        act = torch.zeros_like(distance_squares)
        act[torch.where(distance_squares==0)] = 1
        lr_act = (lr*act).unsqueeze(1)
        delta = lr_act*(input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0)
        delta.div_(input.size(0))
        self.weight.data.add_(delta)
        return loss

    def plot_point_map(self, data, targets, targetnames, filename=None, colors=None, markers=None, example_dict=None,
                       density=True, activities=None):
        """ Visualize the som with all data as points around the neurons
        :param data: {numpy.ndarray} data to visualize with the SOM
        :param targets: {list/array} array of target classes (0 to len(targetnames)) corresponding to data
        :param targetnames: {list/array} names describing the target classes given in targets
        :param filename: {str} optional, if given, the plot is saved to this location
        :param colors: {list/array} optional, if given, different classes are colored in these colors
        :param markers: {list/array} optional, if given, different classes are visualized with these markers
        :param example_dict: {dict} dictionary containing names of examples as keys and corresponding descriptor values
            as values. These examples will be mapped onto the density map and marked
        :param density: {bool} whether to plot the density map with winner neuron counts in the background
        :param activities: {list/array} list of activities (e.g. IC50 values) to use for coloring the points
            accordingly; high values will appear in blue, low values in green
        :return: plot shown or saved if a filename is given
        """
        bmu_locations, loss = self.forward(data.cuda())
        if not markers:
            markers = ['o'] * len(targetnames)
        if not colors:
            colors = ['#00ffaa','#003cff', '#90C3EC', '#EDB233',  '#C02942', '#79BD9A', '#774F38', 'gray', 'black']
        if activities:
            heatmap = plt.get_cmap('coolwarm').reversed()
            colors = [heatmap(a / max(activities)) for a in activities]
        if density:
            fig, ax, wm = self.plot_density_map(data, internal=True)
        else:
            fig, ax = plt.subplots(figsize=(4,4))
        cm_1 = np.zeros((4,4))
        cm_2 = np.zeros((4,4))
        for cnt, xx in enumerate(data):
            if activities:
                c = colors[cnt]
            else:
                c = colors[targets[cnt] if targets[cnt]>0 else 0]
            w = bmu_locations[cnt].squeeze().long().cpu().numpy()
            [cx,cy] = w
            if targets[cnt]>0:
                cm_1[cx,cy] +=1
            else:
                cm_2[cx,cy] +=1
            ax.plot(w[1] + .5 + 0.1 * np.random.randn(1), w[0] + .5 + 0.1 * np.random.randn(1),
                    markers[targets[cnt]], color=c, markersize=12, alpha=0.6)

        ax.set_aspect('equal')
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 4])
        plt.xticks(np.arange(.5, 4 + .5), range(4))
        plt.yticks(np.arange(.5, 4 + .5), range(4))
        ax.grid(which='both')

        if not activities:
            patches = [mptchs.Patch(color=colors[i], label=targetnames[i]) for i in range(len(targetnames))]
            legend = plt.legend(handles=patches, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(targetnames),
                                mode="expand", borderaxespad=0.1)
            legend.get_frame().set_facecolor('#e5e5e5')

        if example_dict:
            for k, v in example_dict.items():
                w = bmu_locations(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

        if filename:
            plt.savefig(filename)
            plt.close()
            print("Point map plot done!")
        else:
            plt.show()
            return wm,cm_1,cm_2
    def plot_density_map(self, data, colormap='Oranges', filename=None, example_dict=None, internal=False):
        """ Visualize the data density in different areas of the SOM.
        :param data: {numpy.ndarray} data to visualize the SOM density (number of times a neuron was winner)
        :param colormap: {str} colormap to use, select from matplolib sequential colormaps
        :param filename: {str} optional, if given, the plot is saved to this location
        :param example_dict: {dict} dictionary containing names of examples as keys and corresponding descriptor values
            as values. These examples will be mapped onto the density map and marked
        :param internal: {bool} if True, the current plot will stay open to be used for other plot functions
        :return: plot shown or saved if a filename is given
        """
        bmu_locations, loss = self.forward(data.cuda())
        wm = np.zeros((4,4), dtype=int)
        for d in range(len(data)):
            [x, y] = bmu_locations[d].squeeze().long().cpu().numpy()
            wm[x, y] += 1
        fig, ax = plt.subplots(figsize=(4,4))
        plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, 4 + .5), range(4))
        plt.yticks(np.arange(.5, 4 + .5), range(4))
        ax.set_aspect('equal')

        if example_dict:
            for k, v in example_dict.items():
                w = bmu_locations[v].squeeze().long().cpu().numpy()
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

        if not internal:
            if filename:
                plt.savefig(filename)
                plt.close()
                print("Density map plot done!")
            else:
                plt.show()
        else:
            return fig, ax, wm