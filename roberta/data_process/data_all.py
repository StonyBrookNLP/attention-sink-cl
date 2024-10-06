import codecs
import os
import io
import numpy as np
import os

def loadFile(fpath):
    with codecs.open(fpath, 'rb', 'utf-8') as f:
        return [line.rstrip() for line in f.read().splitlines()]

def train_test_data(taskpath, data_class_to_global_label, ratio_train = 1, shuffle_train = False):
    train1 = loadFile(os.path.join(taskpath, 's1.train'))
    train2 = loadFile(os.path.join(taskpath, 's2.train'))
    trainlabels = io.open(os.path.join(taskpath, 'labels.train'),
                          encoding='utf-8').read().splitlines()
    trainlabels = np.array([data_class_to_global_label[y] for y in trainlabels], dtype = object)

    if os.path.isfile(os.path.join(taskpath, 's1.dev')): #For GLUE data
        valid1 = loadFile(os.path.join(taskpath, 's1.dev'))
        valid2 = loadFile(os.path.join(taskpath, 's2.dev'))
        validlabels = io.open(os.path.join(taskpath, 'labels.dev'),
                            encoding='utf-8').read().splitlines()
    else:
        valid1 = loadFile(os.path.join(taskpath, 's1.test'))
        valid2 = loadFile(os.path.join(taskpath, 's2.test'))
        validlabels = io.open(os.path.join(taskpath, 'labels.test'),
                            encoding='utf-8').read().splitlines()
    validlabels = np.array([data_class_to_global_label[y] for y in validlabels], dtype = object)

    if shuffle_train:
        index_train = np.random.permutation(len(train1)) 
    else:
        index_train = np.arange(len(train1))  

    #train_length = int(ratio_train * len(index_train))
    train_length = 1245 * len(data_class_to_global_label)
    index_test = np.random.permutation(len(valid1))
    test_length = 113 * len(data_class_to_global_label)
    data = {'train': (np.array(train1, dtype=object)[index_train[:train_length]].tolist(), np.array(train2, dtype=object)[index_train[:train_length]].tolist(), trainlabels[index_train[:train_length]].tolist()),
            'valid': (valid1, valid2, validlabels.tolist()),
            'test': (np.array(valid1, dtype=object)[index_test[:test_length]].tolist(), np.array(valid2, dtype=object)[index_test[:test_length]].tolist(), validlabels[index_test[:test_length]].tolist()),   #used only for MBPA.
            'label_dic': data_class_to_global_label
            }     
    return data

def get_dictionary(word_to_global_label, data_class_to_word):
    #data_class_to_word: {'class in the original data': 'label word'} #given by data
    #Update the dictionary word_to_global_label: {'label word': global label index}
    #Get the dictionary mapping class in the original data to the global label index: data_class_to_global_label: {'class in the original data': global label index} 
    data_class_to_global_label = {}
    #Update the overall dictionary
    for key, value in data_class_to_word.items():
        # The local label word does not appear before
        if value not in word_to_global_label.keys():
            index = len(word_to_global_label.keys())
            word_to_global_label[value] = index
        else:
            index = word_to_global_label[value]
        
    for key, value in data_class_to_word.items():
        data_class_to_global_label[key] = [word_to_global_label[value]]
    return data_class_to_global_label, word_to_global_label

def data_yahoo(taskpath, ratio_train = 1, shuffle_train = True, word_to_global_label = {}, sublabel = None):
    if sublabel == '1':
        data_class_to_word = {'10': 'Politics',  '7': 'Business'}
    elif sublabel == '2':
        data_class_to_word = {'2': 'Science',  '5': 'Computers'}
    elif sublabel == '3':
        data_class_to_word = {'4': 'Education',  '6': 'Sports'}
    elif sublabel == '4':
        data_class_to_word = {'3': 'Health',  '1': 'Society'}
    elif sublabel == '5':
        data_class_to_word = {'9': 'Family',  '8': 'Entertainment'}
    else:
        data_class_to_word = {'1': 'Society',  '2': 'Science', '3': 'Health', '4': 'Education', '5': 'Computers',  '6': 'Sports', '7': 'Business', '8': 'Entertainment', '9': 'Family', '10': 'Politics'}
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label
    
def data_db(taskpath, ratio_train = 1, shuffle_train = True, word_to_global_label = {}, sublabel = None):
    #convert labels
    if sublabel == '1':
        data_class_to_word = {'7': 'Building',  '8': 'Nature'}
    elif sublabel == '2':
        data_class_to_word = {'14': 'Writing',  '13': 'Film'}
    elif sublabel == '3':
        data_class_to_word = {'11': 'Plant',  '1': 'Company'}
    elif sublabel == '4':
        data_class_to_word = {'2': 'Education',  '12': 'Album'}
    elif sublabel == '5':
        data_class_to_word = {'4': 'Athlete',  '9': 'Village'}
    elif sublabel == '6':
        data_class_to_word = {'3': 'Artist',  '10': 'Animal'}
    elif sublabel == '7':
        data_class_to_word = {'5': 'Office',  '6': 'Transportation'}
    else:
        data_class_to_word = {'1': 'Company',  '2': 'Education', '3': 'Artist', '4': 'Athlete', '5': 'Office',  '6': 'Transportation', '7': 'Building', '8': 'Nature', '9': 'Village', '10': 'Animal', '11': 'Plant', '12': 'Album', '13': 'Film', '14': 'Writing'}
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label
    
def data_ag(taskpath, ratio_train = 1, shuffle_train = True, word_to_global_label = {}, sublabel = None):
    #convert labels
    if sublabel == '1':
        data_class_to_word = {'2': 'Sports',  '4': 'Sci/Tech'}
    elif sublabel == '2':
        data_class_to_word = {'3': 'Business',  '1': 'World'}
    else:
        data_class_to_word = {'1': 'World',  '2': 'Sports', '3': 'Business', '4': 'Sci/Tech'}
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label
           
def data_mnli(taskpath, ratio_train = 1, shuffle_train = True, word_to_global_label = {}):
    #convert labels
    data_class_to_word = {'entailment': 'entailment',  'neutral': 'neutral', 'contradiction': 'contradiction'}
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label
        
def data_rte(taskpath, ratio_train = 1, shuffle_train = True, word_to_global_label = {}):
    #convert labels
    data_class_to_word = {'entailment': 'entailment',  'not_entailment': 'non-entailment'}
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label
    
def data_qqp(taskpath, ratio_train = 1, shuffle_train = True, word_to_global_label = {}):
    #convert labels
    data_class_to_word = {'0': 'not duplicate',  '1': 'duplicate'}
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label
    
def data_mrpc(taskpath, ratio_train = 1.0, shuffle_train = True, word_to_global_label = {}):
    #convert labels
    data_class_to_word = {'0': 'paraphrase',  '1': 'not paraphrase'}
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label
    
def data_sst(taskpath, ratio_train = 1, shuffle_train = True, word_to_global_label = {}):
    #convert labels
    data_class_to_word = {'0': 'negative',  '1': 'positive'}
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label

def data_amazon(taskpath, ratio_train = 1, shuffle_train = True, word_to_global_label = {}):
    #convert labels
    data_class_to_word = {'1': '1',  '2': '2', '3': '3', '4': '4', '5': '5'} #Scores
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label

def data_yelp(taskpath, ratio_train = 1, shuffle_train = True, word_to_global_label = {}):
    data_class_to_word = {'1': '1',  '2': '2', '3': '3', '4': '4', '5': '5'} #Scores
    data_class_to_global_label, word_to_global_label = get_dictionary(word_to_global_label, data_class_to_word)
    data = train_test_data(taskpath, data_class_to_global_label, ratio_train, shuffle_train)
    return data, word_to_global_label