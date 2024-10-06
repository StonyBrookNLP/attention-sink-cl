import torch
import numpy as np
from utils.batchify import batchify

def evaluate_class(local_class, model, sentence, segment, target_mask, label, batch_size, device = "cuda", common_mask = None, type_classifier = 'original'):
    model.eval()
    correct = 0
    correct_cil = 0
    rep = []
    
    for i in range(0, len(sentence), batch_size):
        batch_sentence, batch_segment, batch_mask, batch_label, batch_target_mask, batch_common_mask = batchify(sentence[i:i + batch_size], segment[i:i + batch_size], target_mask[i:i + batch_size], label[i:i + batch_size], common_mask[i:i + batch_size])
        batch_sentence, batch_segment, batch_mask, batch_label, batch_target_mask = batch_sentence.to(device), batch_segment.to(device), batch_mask.to(device), batch_label.to(device), batch_target_mask.to(device) 
        if common_mask is not None:
            batch_common_mask = batch_common_mask.to(device)
        
        with torch.no_grad():
            #######Mask out common tokens########
            #batch_mask = (batch_mask - batch_common_mask) * batch_mask
            x, att_all, avg_cos_all, sink_all = model(batch_sentence, batch_mask, batch_segment, batch_common_mask, evaluate = True)
            sink_dev_all, num_common_all = sink_all
            if i == 0:
                over_smooth = np.array(avg_cos_all)
                num_common = np.array(num_common_all)
                sink_dev = np.array(sink_dev_all)
            else:
                over_smooth += np.array(avg_cos_all)
                num_common += np.array(num_common_all)
                sink_dev += np.array(sink_dev_all)
            #Original: batch_mask 
            #Classify without sink token: batch_mask = (batch_mask - batch_common_mask) * batch_mask
            predict_label = model.classify(x, batch_target_mask, batch_mask, local_class, type_classifier)
            predict_label_classil = model.classify(x, batch_target_mask, batch_mask, None, type_classifier)
            
            correct_til_tmp = predict_label.data.max(2)[1].long().eq(batch_label.data.long())
            correct_cil_tmp = predict_label_classil.data.max(2)[1].long().eq(batch_label.data.long())
            correct += correct_til_tmp.sum().cpu().numpy()
            correct_cil += correct_cil_tmp.sum().cpu().numpy()
                
            #Normalize x_cls
            x_cls = torch.sum(x * (batch_target_mask).unsqueeze(2).expand_as(x), dim = 1, keepdim=True)/torch.sum(batch_target_mask, dim = -1, keepdim = True).unsqueeze(2)
            x_norm = torch.sqrt(torch.sum(x_cls ** 2, dim=-1, keepdim=True))
            x_cls = x_cls/(x_norm + 1e-9)
            rep.extend(x_cls.squeeze(1).data.cpu().numpy())
    return correct/len(sentence)*100, correct_cil/len(sentence)*100, rep, over_smooth/len(sentence), (num_common/len(sentence), sink_dev/len(sentence))