import codecs
import string
import os
import io
import csv
import numpy as np

def Style_transfer(fpath, no_sample):
    with open(fpath) as f:
        numline = len(f.readlines())
        index = np.random.permutation(numline)[:no_sample]
    f.close()
    with open(fpath) as f:
        s1 = []
        s2 = []
        labels = []
        spamreader = csv.reader(f, delimiter=',')
        for segments in spamreader:
            s1.append(segments[1])
            s2.append(segments[3])
            labels.append(segments[0])
    f.close()
    s1 = np.array(s1, dtype=object)[index].tolist()
    s2 = np.array(s2, dtype=object)[index].tolist()
    labels = np.array(labels, dtype=object)[index].tolist()
    return s1, s2, labels
        
def Write(s1, s2, labels, ftype):
    for fname in ['s1', 's2', 'labels']:
        fpath = fname + '.' + ftype
        with open(fpath, "w") as f:
            if fname == 's1':
                for i in range(len(s1)):
                    f.write(s1[i] + '\n')
            elif fname == 's2':
                for i in range(len(s2)):
                    f.write(s2[i] + '\n')
            else:
                for i in range(len(labels)):
                    f.write(labels[i] + '\n')
            f.close() 

def Write_split(s1, s2, labels, ftype, sublabel, target_labels, subsample = False):
    dic = f'../Yahoo_{sublabel}/'#f'../Yahoo10000_{sublabel}/'
    if not os.path.exists(dic):
        os.makedirs(dic)
    index = []
    no_class = 1245#int(3668/2)#10000#1245#10000
    cur_no = {target_labels[0]:0, target_labels[1]:0}
    for i in range(len(s1)):
        cur_class = int(labels[i])
        if cur_class in target_labels:
            if subsample:
                if cur_no[cur_class] < no_class:
                    index.append(i)
                    cur_no[cur_class] += 1
            else:
                index.append(i)
                
    for fname in ['s1', 's2', 'labels']:
        fpath = dic + fname + '.' + ftype
        with open(fpath, "w") as f:
            for i in index:
                if fname == 's1':
                    f.write(s1[i] + '\n')
                elif fname == 's2':
                    f.write(s2[i] + '\n')
                else:
                    f.write(labels[i] + '\n')
            f.close()     
                     
def Process(fpath_all):
    no_train = 115000
    no_test = 7600
    for fpath in fpath_all:
        if 'train' in fpath:
            s1, s2, labels = Style_transfer(fpath, no_train)
            ftype = 'train'
        else:
            s1, s2, labels = Style_transfer(fpath, no_test)
            ftype = 'test'
        Write(s1, s2, labels, ftype)
        
def Process_split(fpath_all):
    no_train = 115000
    no_test = 100000
    target_split = [1, 2, 3, 4, 5]
    target_index = [10, 7, 2, 5, 4, 6, 3, 1, 9, 8]
    print(target_index)
    for fpath in fpath_all:
        if 'train' in fpath:
            s1, s2, labels = Style_transfer(fpath, no_train)
            ftype = 'train'
        else:
            s1, s2, labels = Style_transfer(fpath, no_test)
            ftype = 'test' 
        for sublabel in target_split:
            split_index = [sublabel * 2 - 1, sublabel * 2]
            target_labels = [target_index[split_index[0]-1], target_index[split_index[1]-1]]
            if ftype == 'train':
                subsample = True
            else:
                subsample = False
            Write_split(s1, s2, labels, ftype, sublabel, target_labels, subsample)
        
if __name__ == '__main__':
    fpath_all = ['train.csv', 'test.csv']
    #Process(fpath_all)
    Process_split(fpath_all)
    