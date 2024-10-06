from transformers import RobertaTokenizer
import numpy as np

def tokenize_bert(s1, s2, tokenizer_name, max_len = 512): 
    sentence = []
    target_mask = []
    common_mask = []
    common_token = []
    common_untokenized = []
    assert 'roberta' in tokenizer_name
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name) 
    vocabulary = len(tokenizer)

    cls = ["<s>"]
    sep = ["</s>"]
    mask = ["<mask>"]
    pad = ["<pad>"]
    spec_token = {'pad_token': tokenizer.convert_tokens_to_ids(pad)[0], 'mask_token': tokenizer.convert_tokens_to_ids(mask)[0], 'sep_token': tokenizer.convert_tokens_to_ids(sep)[0], 'cls_token': tokenizer.convert_tokens_to_ids(cls)[0]}
    
    #Add common token
    for key, value in spec_token.items():
        if key != 'pad_token': #and key != 'cls_token':
            common_token.append(value)
            common_untokenized.append(key)
    for key in ['.']: #[',', '.', 'The', 'the', 'and', 'And', 'to', 'of', 'for']:#
        common_token.append(tokenizer.convert_tokens_to_ids(key))
        common_untokenized.append(key)
     
    for i in range(len(s1)):
        tmp_s1 = tokenizer.tokenize(s1[i])
        if s2 is not None:
            tmp_s2 = tokenizer.tokenize(s2[i])
            #If exceed the length limit, discard the longer sentence.
            if len(tmp_s1) + len(tmp_s2) > max_len-3:
                if len(tmp_s1) > len(tmp_s2):
                    tokenized_s1 = tmp_s1[:(max_len-len(tmp_s2)-3)]
                    tokenized_s2 = tmp_s2
                else:
                    tokenized_s2 = tmp_s2[:(max_len-len(tmp_s1)-3)]
                    tokenized_s1 = tmp_s1
            else:
                tokenized_s1 = tmp_s1
                tokenized_s2 = tmp_s2 
            # [CLS] s1 [SEP] s2 [SEP]
            sentence_i = cls + tokenized_s1 + sep + tokenized_s2 + sep
        else:
            # [CLS] s1 [SEP]
            tokenized_s1 = tmp_s1[:max_len-2]
            sentence_i = cls + tokenized_s1 + sep
        sentence_tmp = tokenizer.convert_tokens_to_ids(sentence_i)
        sentence.append(sentence_tmp)

        #The target token for prediction (default: cls)
        target_sentence = np.zeros(len(sentence_i))
        target_sentence[0] = 1
        target_mask.append(target_sentence)

        #Mask for the sink token
        common_mask_tmp = np.zeros(len(sentence_tmp))
        for j in range(len(sentence_tmp)):
            if sentence_tmp[j] in common_token or j == 2: #Add the second token to the common token(count except the start/cls token)
                common_mask_tmp[j] = 1
        common_mask.append(common_mask_tmp)
    return sentence, target_mask, spec_token, common_mask, {'index':np.array(common_token), 'token':common_untokenized}, vocabulary