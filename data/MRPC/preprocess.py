import codecs
import string
import os
import io

def Split(fpath, dev_id_path):
    with codecs.open(dev_id_path, 'rb', 'utf-8') as f:
        dev_id = []
        for line in f.read().splitlines():
            segments = line.rstrip().split('\t')
            dev_id.append(segments[0])
    f.close()
        
    with codecs.open(fpath, 'rb', 'utf-8') as f:
        s1_train = []
        s2_train = []
        labels_train = []
        s1_dev = []
        s2_dev = []
        labels_dev = []
        i = 0
        for line in f.read().splitlines():
            i += 1
            if i > 1:
              segments = line.rstrip().split('\t')
              if segments[1] in dev_id:
                  s1_dev.append(segments[3])
                  s2_dev.append(segments[4])
                  labels_dev.append(segments[0])
              else:
                  s1_train.append(segments[3])
                  s2_train.append(segments[4])
                  labels_train.append(segments[0])
    f.close()
    
    fname = ['train.tsv', 'dev.tsv']
    for name in fname:
        with open(name, "w") as f:
            f.write("Quality	#1 String	#2 String\n")
            if 'train' in name:
                for i in range(len(s1_train)):
                    f.write(labels_train[i] + '\t' + s1_train[i] + '\t' + s2_train[i] + '\n')
            else:
                for i in range(len(s1_dev)):
                    f.write(labels_dev[i] + '\t' + s1_dev[i] + '\t' + s2_dev[i] + '\n')
        f.close()
              
def Style_transfer(fpath):
    with codecs.open(fpath, 'rb', 'utf-8') as f:
        s1 = []
        s2 = []
        labels = []
        i = 0
        for line in f.read().splitlines():
            i += 1
            if i > 1:
              segments = line.rstrip().split('\t')
              if 'test' in fpath:
                s1.append(segments[3])
                s2.append(segments[4])
              else:
                s1.append(segments[1])
                s2.append(segments[2])
              '''
              if 'test' in fpath:
                  labels.append('0')
              else:
              '''
              labels.append(segments[0])
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
                    f.write(str(labels[i]) + '\n')
        f.close() 
            
def Process(fpath_all):
    for fpath in fpath_all:
        s1, s2, labels = Style_transfer(fpath)
        if 'train' in fpath:
            ftype = 'train'
        elif 'test' in fpath:
            ftype = 'test'
        else:
            ftype = 'dev'
        Write(s1, s2, labels, ftype)
        
if __name__ == '__main__':
    #Write train/dev.tsv
    Split('train_all.tsv', 'mrpc_dev_id.txt')
    fpath_all = ['train.tsv', 'dev.tsv', 'test.tsv']
    Process(fpath_all)
    