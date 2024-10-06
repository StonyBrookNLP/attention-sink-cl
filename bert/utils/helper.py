import numpy as np
import random
import torch
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import os 
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed, cuda = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_optimizer(model, lr, training_steps):
    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'layernorm.bias', 'layernorm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if ((not any(nd in n for nd in no_decay) and p.requires_grad))], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if ((any(nd in n for nd in no_decay) and p.requires_grad))], 'weight_decay': 0.00}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = lr, eps = 1e-6, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*training_steps), training_steps)
    return optimizer, scheduler

def visualize(x, label, filename, marker_style = 'o', edge_c = 'face'):
    n_class = 0
    for i in label:
        if i > n_class:
            n_class = i
    n_class = n_class + 1
    cmap = plt.cm.hsv(np.linspace(0, 0.9, n_class))
    rep = TSNE(n_components = 2, random_state = 1234).fit_transform(x)
    plt.scatter(x = rep[:,0], y = rep[:,1], marker=marker_style, edgecolors = edge_c, s = 1, c = [cmap[int(l)] for l in label])
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 5)
    #plt.set_size_inches(18.5, 10.5)
    plt.savefig(filename)
    plt.clf()

def distribution_plot(corr, num_bin = 100, filename = 'model_output/frequent_corr_distribution.jpg'):
    #sort the corr list: 
    sorted_corr = torch.sort(corr, dim=-1, descending=False)[0]
    sorted_corr = sorted_corr.data.cpu().numpy()
    max_corr, min_corr = sorted_corr[-1], sorted_corr[0]
    num_bin = 100
    interval = (max_corr - min_corr)/(num_bin - 1)
    frequency = np.zeros(num_bin, dtype = int)
    corr_axis = np.zeros(num_bin)
    for i in range(num_bin):
        corr_axis[i] = min_corr + i * interval
    for i in range(len(sorted_corr)):
        bin_index = int((corr[i] - min_corr)/interval)
        frequency[bin_index] += 1
    #Plot the frequency
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(corr_axis, frequency, c='b', linewidth=3.0)  
    plt.savefig(filename)
    plt.clf()
