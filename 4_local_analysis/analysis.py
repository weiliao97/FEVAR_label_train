# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 22:37:46 2022

@author: 320190618
"""
import numpy as np
import sklearn.metrics as sk
true = np.asarray([2, 0, 1, 3, 1, 0, 1, 0, 1, 1, 1, 2, 3, 1, 2, 0, 3, 2, 1, 0, 1, 3, 0, 3,
        2, 0, 1])
pred = np.asarray([[2],
        [2],
        [1],
        [3],
        [1],
        [0],
        [1],
        [2],
        [1],
        [1],
        [1],
        [2],
        [3],
        [1],
        [2],
        [3],
        [3],
        [2],
        [1],
        [0],
        [2],
        [3],
        [0],
        [2],
        [2],
        [0],
        [1]],
     )

#%%
#plot cm matrix 
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
matplotlib.rcParams["figure.dpi"] = 300
plt.style.use('bmh')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
legend_properties = {'weight':'bold', 'size': 10}
cm = sk.confusion_matrix(true, pred)
num_class = 4
label_x = ['Pred-SMA/Celiac', 'Pred-Renal', 'Pred-Iliac', 'Pred-All']
 # ['Pred_GW', 'Pred_Sheath', 'Pred_Unsheathed', 'Pred_Branch\nstent', 'Pred_Final\ndeployment']
label_y =  ['SMA/Celiac', 'Renal', 'Iliac', 'All']
# ['GW_only', 'Sheath', 'Unsheathed', 'Branch\nstent', 'Final\ndeployment']
cf_matrix = cm/np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)
group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
# percentage based on true label 
gr = (cm/np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)).flatten()
group_percentages = ['{0:.1%}'.format(value) for value in gr]

labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_percentages, group_counts)]

labels = np.asarray(labels).reshape(num_class, num_class)

if label_x is not None:
    xlabel = label_x
    ylabel = label_y
else:
    xlabel = ['Pred-%d'%i for i in range(num_class)]
    ylabel = ['%d'%i for i in range(num_class)]

sns.set(font_scale = 1.5)

hm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap = 'OrRd', \
annot_kws={"fontsize": 16}, xticklabels=xlabel, yticklabels=ylabel, cbar=False)
# hm.set(title=title)
plt.xticks(rotation = 15)
fig = plt.gcf()
plt.show()

#%%
import torch.nn as nn
import torch 
# With square kernels and equal stride
m = nn.Conv2d(3, 3, 3, stride=2)
input = torch.randn(1, 3, 968, 968)
n = nn.MaxPool2d(kernel_size=2)
output = n(m(input))