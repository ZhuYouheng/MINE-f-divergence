import torch
import matplotlib.pyplot as plt
from Gibbs import generate_sw_tuples_batch
from MINE import MINE
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np


model = MINE(6, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
plot_loss = []

import json
with open('data/gibbs_data_p_is_4.json', 'r') as datafile:
    data = json.load(datafile)
S = np.array([data[i]['S'][0] for i in range(len(data))])
w = np.array([data[i]['w'] for i in range(len(data))])
w_shuffle = np.random.permutation(w)
S_sample = Variable(torch.from_numpy(S).type(torch.FloatTensor), requires_grad=True)
w_sample = Variable(torch.from_numpy(w).type(torch.FloatTensor), requires_grad=True)
w_shuffle_sample = Variable(torch.from_numpy(w_shuffle).type(torch.FloatTensor), requires_grad=True)

kl = []
num_epoch = 400
for epoch in tqdm(range(num_epoch)):
    pred_xy = model(S_sample, w_sample)
    pred_x_y = model(S_sample, w_shuffle_sample)
    
    loss1 = - torch.mean(pred_xy)
    loss2 = torch.log(torch.mean(torch.exp(pred_x_y)))
    
    kl.append((-loss1-loss2).data.numpy())
    model.zero_grad()
    loss1.backward()
    optimizer.step()

x_plot = [x for x in range(num_epoch)]
plt.plot(x_plot, kl)