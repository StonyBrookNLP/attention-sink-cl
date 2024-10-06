from utils.tokenize import tokenize_bert
from utils.helper import set_seed

from train.train_cl import train

from data_process.read_data import read_data
from data_process.task_seq import get_subset
import numpy as np
import torch
import model.transformer_model as transformer
from transformers import RobertaForMaskedLM
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--enc_type', type=str, default='roberta-base')
parser.add_argument('--bsize', type=int, default=32)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--sequence', type=str, nargs='+', default=['snli'])
parser.add_argument('--freeze_pretrain', action="store_true", default=False)
parser.add_argument('--freeze_all', action="store_true", default=False)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--plotrep', action="store_true", default=False)
parser.add_argument('--ratio_train', type=float, default=0.1)
parser.add_argument('--type_classifier', type=str, default='prescale')

args = parser.parse_args()

if __name__ == '__main__':
    device = "cuda"
    #Parameters
    seed = args.seed
    encoder_type = args.enc_type 
    sequence = args.sequence   
    freeze_pretrain = args.freeze_pretrain
    freeze_all = args.freeze_all
    max_epoch = args.epoch
    plotrep = args.plotrep
    clr = args.lr
    ratio_train = args.ratio_train
    type_classifier = args.type_classifier

    if 'large' in encoder_type:
        hidden = 1024
        hidden_latent = 1024
        n_head = 16
        n_layer = 24
    else:
        hidden = 768
        hidden_latent = 768
        n_head = 12
        n_layer = 12
    
    set_seed(seed)
    model_type = 'bert'
    dataset_all = [] 
    word_to_global_label = {}
    data_index = np.random.permutation(len(sequence))
    #data_index = np.arange(len(data_all))

    #The model name to save (reflect the sequence information)
    for j in range(len(sequence)):
        dataset = sequence[data_index[j]]
        if j == 0:
            save_name = dataset
        else:
            save_name = save_name + '_' + dataset
        print("==============================") 
        print(dataset)
        print("==============================")
        
        subset = get_subset(dataset)
        
        #Shuffle the order of the subsets
        if subset[0] is not None:
            subset_index = np.random.permutation(len(subset))
            subset = np.array(subset, dtype = object)[subset_index].tolist()
                
        for sub in subset:
            print("==============================") 
            print(sub)
            print("==============================")
            data, batch_size, n_class, max_len, word_to_global_label = read_data(dataset, sub, args, word_to_global_label, ratio_train = ratio_train)
            print(word_to_global_label)
            
            sent_train, target_mask_train, spec_token, common_mask_train, common_token, vocabulary = tokenize_bert(data['train'][0], data['train'][1], encoder_type, max_len)
            label_train = data['train'][2]
                        
            sent_valid, target_mask_eval, _, common_mask_eval, _, _ = tokenize_bert(data['valid'][0], data['valid'][1], encoder_type, max_len)
            label_valid = data['valid'][2]
            
            sent_test, target_mask_test, _, common_mask_test, _, _ = tokenize_bert(data['test'][0], data['test'][1], encoder_type, max_len)
            label_test = data['test'][2]
            
            if sub is not None:
                name = dataset + sub
            else:
                name = dataset
            dataset_all.append({'train':(sent_train, target_mask_train, (np.array(label_train)).tolist(), common_mask_train),
                              'dev':(sent_valid, target_mask_eval, (np.array(label_valid)).tolist(), common_mask_eval),
                              'test':(sent_test, target_mask_test, (np.array(label_test)).tolist(), common_mask_test),
                              'local_dic':data['label_dic'],
                              'name':name}) 
            
    #Configuration of task-wise class mask
    class_all = len(word_to_global_label .keys())
    for i in range(len(dataset_all)):
        #all_class masks all classes seen before current task
        if i == 0:
            all_class = np.zeros(class_all)
        local_class = np.zeros(class_all)
        data_class_to_global_label = dataset_all[i]['local_dic']
        for key, value in data_class_to_global_label.items():
            local_class[value] = 1
        print('local class:')
        print(local_class)
        all_class = all_class + local_class * (1 - all_class)
        print('all class')
        print(all_class)
        #local_class = all_class
        dataset_all[i]['class'] = torch.Tensor(local_class).long().unsqueeze(0) #size: 1, n_class
        dataset_all[i]['all_class'] = torch.Tensor(all_class).long().unsqueeze(0) #size: 1, n_class
    #mask_token, cls_token, sep_token, pad_token = spec_token['mask_token'], spec_token['cls_token'], spec_token['sep_token'], spec_token['pad_token']
    kwargs = {'encoder_type': encoder_type, 'n_embd': hidden, 'n_layer': n_layer, 'n_head': n_head, "spec_token":spec_token}
    config = transformer.TransformerConfig(vocabulary, class_all, kwargs)

    #Load the pre-trained encoder        
    LM_pretrained = RobertaForMaskedLM.from_pretrained(encoder_type)
    model = transformer.Transformer(config, class_head = True) 
    model.load_from_roberta(LM_pretrained.roberta)
    del LM_pretrained
    if torch.cuda.is_available():
        model = model.cuda()
    
    print ("[Tran] number of parameters in encoder: ") 
    num = 0   
    for n, p in list(model.named_parameters()):
        if ('classifier' not in n) and ('pooler' not in n):
            if p.requires_grad:
                num += p.numel()
    print(num)
    
    print("TRAINING")
    train(max_epoch, clr, model, dataset_all, batch_size, device, save_name, type_classifier, plotrep) 