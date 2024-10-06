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
            s2.append(segments[2])
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
        
if __name__ == '__main__':
    fpath_all = ['train.csv', 'test.csv']
    Process(fpath_all)
    