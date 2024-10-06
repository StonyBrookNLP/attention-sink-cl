import torch
import torch.nn as nn
from model.classifier import Classifier
import torch.nn.functional as F 
import logging 
import math
logger = logging.getLogger(__name__)

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)
        
def stable_softmax(att, local_class = None):
    if local_class is not None:
        local_class = local_class.unsqueeze(-2).expand_as(att)
        att = att.masked_fill(local_class == 0, -1e9) 
    att_max = torch.max(att, dim = -1, keepdim = True)[0]
    att = att - att_max
    att = torch.exp(att)
    att = att/torch.sum(att, dim = -1, keepdim = True)  
    return att

class TransformerConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_layer = 12
    n_head = 12
    n_embd = 768
    max_position_embeddings = 514
    type_vocab_size = 2
    
    def __init__(self, vocab_size, n_class, kwargs):
        self.vocab_size = vocab_size
        self.n_class = n_class
        for k,v in kwargs.items():
            setattr(self, k, v)        
               
class SelfAttention(nn.Module):

    def __init__(self, config):
        super(SelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # drop out
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.layernorm = nn.LayerNorm(config.n_embd, eps = 1e-12)
    
    def self_att(self, x, mask):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x)
        q = self.query(x) 
        v = self.value(x)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))   
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, x.size(1), 1)
            mask = mask.unsqueeze(1)
            att = att.masked_fill(mask == 0, -1e9)   

        att = stable_softmax(att)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)    
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y, att # B, nh, T, T

    def forward(self, x, mask, common_mask, evaluate = False): 
        y, att = self.self_att(x, mask)
        y = self.layernorm(y + x)    
        if evaluate:
            #################################################################################################################################
            #We use k=1 when calculating the ratio of sink token that are also common tokens and sink att deviation, and use k=5 for overall (sink) att dev estimation in experiments section
            #################################################################################################################################
            k=1
            mask_largest_degree, sink_dev = self.avg_degree_dev(att, k, mask)
            #Sink token deviation
            sink_dev = torch.sum(torch.sum(sink_dev, dim = -1))/(self.n_head * k)
            #number of tokens that are in the common tokens
            num_common = torch.sum(mask_largest_degree * common_mask.unsqueeze(1))/(self.n_head * k)
        else:
           sink_dev, num_common = None, None
        return y, att, sink_dev, num_common 
    
    def avg_degree_dev(self, attn, k, mask):
        B, nh, T = attn.size(0), attn.size(1), attn.size(2)
        #Calculate the average outer degree
        attn = attn * mask.unsqueeze(1).unsqueeze(3) #B, nh, T, T
        degree = torch.sum(attn, dim = -2, keepdim = True)#B, nh, 1, T
        avg_degree = degree/(torch.sum(mask, dim = -1, keepdim = True).unsqueeze(1).unsqueeze(3)) #B, nh, 1, T
        degree_list, degree_index_origin = torch.sort(avg_degree, dim = -1, descending = True)
        largest_index = degree_index_origin[:, :, 0, :k] #B, nh, k

        #Mask the tokens with largest outer degrees
        mask_largest_degree = torch.zeros(B, T).to(attn.device)
        mask_largest_degree = mask_largest_degree.unsqueeze(1).repeat(1, nh, 1) #B, nh, T
        mask_largest_degree.scatter_(-1, largest_index, 1.)

        #Calculate the attention deviation
        sink_dev = ((attn-avg_degree) ** 2) * mask.unsqueeze(1).unsqueeze(3) #B, nh, T, T
        sink_dev = torch.sqrt(torch.sum(sink_dev, dim = -2, keepdim = True))/(degree + 1e-9)
        sink_dev = sink_dev.squeeze(2) * mask_largest_degree * mask.unsqueeze(1)#/k #B, T
        return mask_largest_degree, sink_dev

    def load_from_roberta(self, roberta_attention):
        self.key.weight.data.copy_(roberta_attention.self.key.weight.data)
        self.key.bias.data.copy_(roberta_attention.self.key.bias.data)

        self.query.weight.data.copy_(roberta_attention.self.query.weight.data)
        self.query.bias.data.copy_(roberta_attention.self.query.bias.data)

        self.value.weight.data.copy_(roberta_attention.self.value.weight.data)
        self.value.bias.data.copy_(roberta_attention.self.value.bias.data)

        self.proj.weight.data.copy_(roberta_attention.output.dense.weight.data)
        self.proj.bias.data.copy_(roberta_attention.output.dense.bias.data)

        self.layernorm.weight.data.copy_(roberta_attention.output.LayerNorm.weight.data) 
        self.layernorm.bias.data.copy_(roberta_attention.output.LayerNorm.bias.data) 
            
    def freeze_pretrain(self):
        for n, p in list(self.key.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.query.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.value.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.proj.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.layernorm.named_parameters()):
            p.requires_grad = False 

    '''
    #Check practical value for over-smoothing inequality
    def check_singular_value(self):
        S = torch.linalg.svdvals(self.value.weight.data)
        max_singular = torch.max(S)
        return max_singular
     
    def check_attn_eigen(self, attn, mask):
        B = attn.size(0)
        eigen_average = 0
        n_all = torch.sum(mask, dim = -1)
        for i in range(B):
            n = int(n_all[i])
            A = attn[i]
            A = A[:, :n, :n] #n, T
            eet = (torch.eye(n) - 1/n * torch.ones([n, n])).to(A.device) #n * n
            eet = eet.unsqueeze(0).repeat(A.size(0), 1, 1)
            A = (A.transpose(-1, -2) @ eet)@A#B, nh, n, n
            L, V = torch.linalg.eig(A)
            L_magnitude_max = (torch.view_as_real(L)[:, :, 0])
            L_magnitude_max = torch.max(L_magnitude_max, dim = -1)[0]
            L_magnitude_max = torch.mean(L_magnitude_max)
            eigen_average = eigen_average + L_magnitude_max
        return eigen_average
    
    def check_std(self, x, mask):
        mean = torch.sum(x, dim = -1, keepdim = True)/x.size(-1) #B, T, 1
        std = torch.sqrt(torch.sum((x - mean) ** 2, dim = -1)/x.size(-1)) #B, T
        if mask is not None:
            std = std.masked_fill(mask == 0, 1e9)   
        std_max = torch.min(std)
        return std_max
    '''
    
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attn = SelfAttention(config)
        self.n_head = config.n_head
        self.intermediate = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU()
        )
        self.output = nn.Sequential(
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
        self.layernorm = nn.LayerNorm(config.n_embd, eps = 1e-12)
        
    def forward(self, x, mask, common_mask, evaluate = False):
        x, att_layer, sink_dev, num_common  = self.attn(x, mask, common_mask, evaluate)
        tmp = x
        tmp = self.intermediate(tmp) 
        tmp = self.output(tmp)
        x = self.layernorm(x + tmp)
        return x, att_layer, sink_dev, num_common

    def cosine_rep(self, x, mask):
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm_matrix = norm @ norm.transpose(1, 2)
        att = x @ x.transpose(-2,-1) / norm_matrix
    
        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(-1) == 0, 0)
            att = att.masked_fill(mask.unsqueeze(1) == 0, 0)
        att = att.unsqueeze(1) #Add the head dimension
        return att
    
    def load_from_roberta(self, roberta_layer):
        self.attn.load_from_roberta(roberta_layer.attention)
        self.intermediate[0].weight.data.copy_(roberta_layer.intermediate.dense.weight.data)
        self.intermediate[0].bias.data.copy_(roberta_layer.intermediate.dense.bias.data)
        self.output[0].weight.data.copy_(roberta_layer.output.dense.weight.data)
        self.output[0].bias.data.copy_(roberta_layer.output.dense.bias.data)
        self.layernorm.weight.data.copy_(roberta_layer.output.LayerNorm.weight.data)
        self.layernorm.bias.data.copy_(roberta_layer.output.LayerNorm.bias.data)
            
    def freeze_pretrain(self):
        self.attn.freeze_pretrain()
        for n, p in list(self.intermediate.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.output.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.layernorm.named_parameters()):
            p.requires_grad = False

class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        self.n_embd = config.n_embd
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.segment_emb = nn.Embedding(config.type_vocab_size, config.n_embd)  
        self.position_emb = nn.Embedding(config.max_position_embeddings, config.n_embd)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer("vocab_ids", torch.arange(config.vocab_size).expand((1, -1)))
        self.layernorm = nn.LayerNorm(config.n_embd, eps = 1e-12)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.spec_token = config.spec_token
    
    def cosine_att(self, x, mask):
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm_matrix = norm @ norm.transpose(1, 2)
        att = x @ x.transpose(-2,-1) / norm_matrix

        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(-1) == 0, 0)
            att = att.masked_fill(mask.unsqueeze(1) == 0, 0)
        att = att.unsqueeze(1) #Add the head dimension
        return att  
                
    def forward(self, x):
        position_ids = self.position_ids[:, :x.size(1)].cuda()
        position = self.position_emb(position_ids)
        token = self.tok_emb(x)

        emb = token + position
        emb = self.layernorm(emb)
        emb = self.drop(emb)
        return emb
         
    def load_from_roberta(self, roberta_emb):
        self.tok_emb.weight.data.copy_(roberta_emb.word_embeddings.weight.data) 
        self.segment_emb.weight.data.copy_(roberta_emb.token_type_embeddings.weight.data) 
        self.position_emb.weight.data.copy_(roberta_emb.position_embeddings.weight.data) 
        self.layernorm.weight.data.copy_(roberta_emb.LayerNorm.weight.data)
        self.layernorm.bias.data.copy_(roberta_emb.LayerNorm.bias.data)
            
    def freeze_pretrain(self):
        for n, p in list(self.tok_emb.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.segment_emb.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.position_emb.named_parameters()):
            p.requires_grad = False
        for n, p in list(self.layernorm.named_parameters()):
            p.requires_grad = False  

    def check_norm(self, rep_matrix = None, indices = None):
        if rep_matrix is None:
            rep_matrix = self.tok_emb.weight.data
        norm = torch.sum(rep_matrix * rep_matrix, dim = -1) #vocab
        return norm
    
    def corr_embd(self, indices, full = True):
        query_matrix = self.tok_emb.weight.data[indices] 
        if full:
            rep_matrix = self.tok_emb.weight.data
        else:
            rep_matrix = self.tok_emb.weight.data[indices] 
        att = query_matrix @ rep_matrix.transpose(0,1)
        return att
    
class Transformer(nn.Module):
    def __init__(self, config, class_head=True):
        super(Transformer, self).__init__()
        # input embedding stem
        self.emb = BertEmbedding(config)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.class_head = class_head
        self.class_no = config.n_class
        # transformer
        self.layer = config.n_layer
        module = []
        for i in range(config.n_layer):
            module.append(Block(config))
        self.blocks = nn.ModuleList(module)
        if class_head:
            self.classifier = Classifier(config)
    
        self.vocab = config.vocab_size 
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
            
    def load_from_roberta(self, roberta_model):
        self.emb.load_from_roberta(roberta_model.embeddings)
        i = 0
        for layer in self.blocks:
            layer.load_from_roberta(roberta_model.encoder.layer[i]) 
            i += 1

    def freeze_pretrain(self):
        self.emb.freeze_pretrain()
        i = 0
        for layer in self.blocks:
            layer.freeze_pretrain()
            i += 1
   
    def finetune_all(self):  
        for n, p in self.named_parameters(): 
            p.requires_grad = True  

    def freeze_classifier(self):
        self.classifier.freeze_parameters()

    def get_config(self):
        return self.config
        
    #seqJoint_model(batch_sentence, batch_mask, batch_segment, batch_inter_mask, batch_prefix_mask, batch_target_mask, cls_no = cls_no)       
    def forward(self, x, mask, common_mask, evaluate = False):
        att_all = []
        avg_cos_all = []
        num_common_all = []
        sink_dev_all = [] 
        #Diagonal matrix to exclude self similarity when calculating the representationial cosine similarity
        diag = 1 - torch.eye(x.size(1)).unsqueeze(0).unsqueeze(1).to(x.device) 

        #Embedding layer
        emb = self.emb(x) 
        x = emb.clone()

        #Attention layer
        i = 0
        for layer in self.blocks:
            x, att_layer, sink_dev, num_common = layer(x, mask, common_mask, evaluate)

            #If evaluate, record the sink dev, common num and representational cosine similarity
            if evaluate:
                #num common token and sink att dev
                num_common_all.append(num_common.data.cpu().numpy())
                sink_dev_all.append(sink_dev.data.cpu().numpy())
                #Cosine similarity
                cos_rep = layer.cosine_rep(x, mask)
                avg_cos = torch.sum((cos_rep * diag).view(x.size(0), -1), dim = -1)/(torch.sum(mask, dim = -1) * (torch.sum(mask, dim = -1) - 1))
                avg_cos = torch.sum(avg_cos)
                avg_cos_all.append(avg_cos.data.cpu().numpy())
                #Raw attention
                att_all.append(att_layer)
        return x, att_all, avg_cos_all, (sink_dev_all, num_common_all)

    def classify(self, x, target_mask = None, mask = None, local_class = None, type_classifier = 'original'): #Classification performs after the decoder
        if type_classifier == 'original':
            class_prob = self.classifier(x, target_mask, local_class) #original classifier
        elif type_classifier == 'uniform':
            class_prob  = self.classifier.attn_classifier_uniform(x, mask, local_class)
        else:
            class_prob = self.classifier.attn_classifier(x, mask, local_class)
        return class_prob         
            
    def class_loss(self, class_predict, target):
        class_prob = class_predict.gather(2, target.unsqueeze(2).repeat(1, class_predict.size(1), 1))#.squeeze(1)
        class_loss = -torch.sum(class_prob)
        class_loss = class_loss / class_predict.size(0)
        return class_loss

    