import numpy as np
import torch

def batchify(sentence, segment, target_mask, labels, common_mask = None):
    lengths = np.array([len(x) for x in sentence])
    n_sent = len(sentence)
    max_len = int(np.max(lengths))
    
    batch_sentence = torch.zeros([n_sent, max_len], dtype = torch.long)
    batch_segment = torch.LongTensor(n_sent, max_len).zero_()
    batch_mask = torch.LongTensor(n_sent, max_len).zero_()
    batch_label = torch.LongTensor(n_sent, 1)
    batch_target_mask = torch.LongTensor(n_sent, max_len).zero_() # Specify the position of [MASK]
    batch_common_mask = torch.LongTensor(n_sent, max_len).zero_() 

    for i in range(n_sent):
        
        sent_len = len(sentence[i])
        batch_sentence[i, :sent_len] = torch.Tensor(sentence[i]).long()
        batch_segment[i, :sent_len] = torch.Tensor(segment[i]).long()
        batch_mask[i, :sent_len] = 1
        batch_label[i, :] = torch.Tensor(labels[i]).long()
        batch_target_mask[i, :sent_len] = torch.Tensor(target_mask[i]).long()

        if common_mask is not None:
            batch_common_mask[i, :sent_len] = torch.Tensor(common_mask[i]).long()
    return batch_sentence, batch_segment, batch_mask, batch_label, batch_target_mask, batch_common_mask 
    