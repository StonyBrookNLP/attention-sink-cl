import torch
import torch.nn as nn
import math
import torch.nn.functional as F 

def stable_softmax(att, local_class = None):
    if local_class is not None:
        local_class = local_class.unsqueeze(-2).expand_as(att)
        att = att.masked_fill(local_class == 0, -1e9) 
    att_max = torch.max(att, dim = -1, keepdim = True)[0]
    att = att - att_max
    att = torch.exp(att)
    att = att/torch.sum(att, dim = -1, keepdim = True)  
    return att

class Classifier(nn.Module):
    def __init__(self, config):       
        super(Classifier, self).__init__() 
        self.classifier = nn.Linear(config.n_embd, config.n_class, bias=True)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.attn_drop = nn.Dropout(config.embd_pdrop)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.key.weight.data.zero_()
        self.key.bias.data.zero_()
    
    def attn_classifier(self, x, mask, local_class = None):
        B, T, C = x.size()
        n_head = 1

        query = self.classifier.weight.unsqueeze(0).repeat(B, 1, 1) #B, n_class, C
        n_class = query.size(1)
        
        q = query
        k = self.key(x) 
        v = x
        
        q = q.view(B, n_class, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))   
        if mask is not None:
            mask = mask.unsqueeze(1)#.repeat(1, x.size(1), 1) #B, 1, 1, T
            mask = mask.unsqueeze(1) #B, 1, T, T
            att = att.masked_fill(mask == 0, -1e9)  
        att = stable_softmax(att) 
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)    
        y = y.transpose(1, 2).contiguous().view(B, n_class, C) # B, n_class, C
        
        class_predict = torch.sum(y * query, dim = -1) + self.classifier.bias.unsqueeze(0)#B, 1, n_class + bias
        class_predict = class_predict.unsqueeze(1)
        if local_class is None:
            class_prob = F.log_softmax(class_predict, dim=2)
        else:
            class_prob = torch.log(stable_softmax(class_predict, local_class) + 1e-9)
        return class_prob
    
    def attn_classifier_uniform(self, x, mask, local_class = None):
        B, T, C = x.size()
        n_head = 1
        query = self.classifier.weight.unsqueeze(0).repeat(B, 1, 1) #B, n_class, C
        n_class = query.size(1)

        v = x.view(B, T, n_head, C // n_head).transpose(1, 2)

        att = torch.ones(B, n_head, n_class, T).to(x.device)
        if mask is not None:
            mask = mask.unsqueeze(1) 
            mask = mask.unsqueeze(1)
            att = att.masked_fill(mask == 0, -1e9)  
            
        att = stable_softmax(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, n_class, C) # B, n_class, C
        class_predict = torch.sum(y * query, dim = -1) + self.classifier.bias.unsqueeze(0)#B, 1, n_class + 
        class_predict = class_predict.unsqueeze(1)
        if local_class is None:
            class_prob = F.log_softmax(class_predict, dim=2)
        else:
            class_prob = torch.log(stable_softmax(class_predict, local_class) + 1e-9)
        return class_prob
    
    def forward(self, x, target_mask = None, local_class = None): 
        if target_mask is None:
            x_mask = x
        else:
            target_mask = target_mask.unsqueeze(2).expand_as(x)
            x_mask = torch.sum(x * target_mask, dim = 1, keepdim=True)/torch.sum(target_mask, dim = 1, keepdim = True)

        class_predict = self.classifier((x_mask))#B, 1, n_class

        if local_class is None:
            # softmax over all classes
            class_prob = F.log_softmax(class_predict, dim=2)
        else:
            # softmax over local classes
            class_prob = torch.log(stable_softmax(class_predict, local_class) + 1e-9)
        return class_prob