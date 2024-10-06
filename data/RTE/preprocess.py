import codecs
import string
import os
import io

def Style_transfer(fpath):
    with codecs.open(fpath, 'rb', 'utf-8') as f:
        s1 = []
        s2 = []
        labels = []
        for line in f.read().splitlines():
            segments = line.rstrip().split('\t')
            s1.append(segments[1])
            s2.append(segments[2])
            if len(segments) == 4:
                labels.append(segments[3])
            else:
                labels.append('not_entailment')
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
    for fpath in fpath_all:
        s1, s2, labels = Style_transfer(fpath)
        if 'train' in fpath:
            ftype = 'train'
        elif 'dev' in fpath:
            ftype = 'dev'
        else:
            ftype = 'test'
        Write(s1, s2, labels, ftype)
        
if __name__ == '__main__':
    fpath_all = ['train.tsv', 'dev.tsv', 'test.tsv']
    Process(fpath_all)
    