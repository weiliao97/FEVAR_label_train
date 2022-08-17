# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:47:57 2022

@author: 320190618
"""

#%%
#plot cm matrix 
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import numpy as np
import sklearn.metrics as sk

matplotlib.rcParams["figure.dpi"] = 300
plt.style.use('bmh')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
legend_properties = {'weight':'bold', 'size': 10}
true = xxx
pred = xxxx
cm = sk.confusion_matrix(true, pred)
num_class = 5
label_x = ['Pred_GW', 'Pred_Sheath', 'Pred_Unsheathed', 'Pred_Branch\nstent', 'Pred_Final\ndeployment']
label_y =  ['GW_only', 'Sheath', 'Unsheathed', 'Branch\nstent', 'Final\ndeployment']
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
plt.xticks(rotation = 45)
fig = plt.gcf()
plt.show()