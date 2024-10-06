import os
from utils.batchify import batchify
import torch
import math
import numpy as np
from train.evaluate import evaluate_class
from utils.helper import visualize, set_optimizer

def train_each_epoch(local_class, optimizer, scheduler, model, sentence, target_mask, label, batch_size, device = "cuda", common_mask = None, type_classifier = 'original'):
    model.train()
    print_each = batch_size * 1000
    
    correct = 0
    loss_all = []

    #Shuffle the sentence
    index = np.random.permutation(len(sentence))
        
    sentence = np.array(sentence, dtype=object)[index].tolist()
    target_mask = np.array(target_mask, dtype=object)[index].tolist()
    label = np.array(label, dtype=object)[index].tolist()
    common_mask = np.array(common_mask, dtype=object)[index].tolist()

    max_data_len = len(sentence)
    for i in range(0, max_data_len, batch_size):  
        batch_sentence, batch_mask, batch_label, batch_target_mask, batch_common_mask = batchify(sentence[i:i + batch_size], target_mask[i:i + batch_size], label[i:i + batch_size], common_mask[i:i + batch_size])
        batch_sentence, batch_mask, batch_label, batch_target_mask = batch_sentence.to(device), batch_mask.to(device), batch_label.to(device), batch_target_mask.to(device)
        if common_mask is not None:
            batch_common_mask = batch_common_mask.to(device)
        #######Mask out common tokens########
        #batch_mask = (batch_mask - batch_common_mask) * batch_mask
       
        x = model(batch_sentence, batch_mask, batch_common_mask)[0]
        class_predict = model.classify(x, batch_target_mask, batch_mask, local_class, type_classifier)
        
        correct += class_predict.data.max(2)[1].long().eq(batch_label.data.long()).sum().cpu().numpy()

        loss = model.class_loss(class_predict, batch_label)
        loss_all.append(loss.data.cpu().numpy())

        if i % print_each == 0:
            print("Sentence: " + str(i) + "; The loss is:" + str(np.sum(loss_all)/len(loss_all)) + "; The classification accuracy is:" + str(correct/(i + len(batch_label))*100)) 
        
        #Optimization
        optimizer.zero_grad() 
        loss.backward()
         
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()       
        if scheduler is not None:   
            scheduler.step()
    return correct/(len(sentence)) * 100

def train_task(max_epoch, lr, model, data_current, batch_size, type_classifier = 'original', device = "cuda", save_name = 'news_series', save_model = False, output_dir = './model_output'): 
    #Set optimizer   
    length_all = len(data_current['train'][0])
    training_steps = int(math.ceil(length_all/batch_size)) * max_epoch
    print('Training steps:'+ str(training_steps))
    optimizer, scheduler = set_optimizer(model, lr, training_steps)

    epoch = 1
    while epoch <= max_epoch:
        print("epoch:" + str(epoch))
        sentence, target_mask, label, common_mask = data_current['train']
        local_class = data_current['class'].to(device)
        
        acc_train = train_each_epoch(local_class, optimizer, scheduler, model, sentence, target_mask, label, batch_size, device, common_mask, type_classifier)
        if epoch == max_epoch: #if epoch == True:
            sentence_val, target_mask_val, label_val, common_mask_eval = data_current['dev']
            local_class = data_current['class'].to(device)
            eval_acc_til, eval_acc_cil, _, _, _ = evaluate_class(local_class, model, sentence_val, target_mask_val, label_val, batch_size, device, common_mask_eval, type_classifier)
            print("The evaluation accuracy is:" + str(eval_acc_til)) 
 
        #Early Stop
        if save_model: 
            print("Saving the model at epoch " + str(epoch))
            model_to_save = model.module if hasattr(model, 'module') else model 
            output_model_file = os.path.join(output_dir, save_name + "_class.pth")
            torch.save(model_to_save.state_dict(), output_model_file)
        
        epoch += 1
    #seqJoint_model = best_model
    return eval_acc_til, eval_acc_cil
    
def train(max_epoch, lr, model, data_all, batch_size, device = "cuda", save_name = 'news_series', type_classifier = 'original', plot_rep = False):
    output_dir = './model_output'

    before = []
    before_cil = []
    after = []
    after_cil = []
    
    print("==============================")
    print("Classification Only Training") 
    print("==============================")

    total_task_num = len(data_all)
    for task in range(len(data_all)):
        print("Task" + str(task))
        data_current = data_all[task]  
        save_model_train = False
       
        print("Classifier Training: Probing")
        #Classifier only training
        model.freeze_pretrain()
        class_lr = 1e-3
        class_max_epoch = 3
        eval_acc_til, eval_acc_cil = train_task(class_max_epoch, class_lr, model, data_current, batch_size, type_classifier, device, save_name, save_model = save_model_train, output_dir = output_dir)
        
        #Fine-tuning
        print("All model Training: Fine-Tuning")
        model.finetune_all()
        eval_acc_til, eval_acc_cil = train_task(max_epoch, lr, model, data_current, batch_size, type_classifier, device, save_name, save_model = save_model_train, output_dir = output_dir)
        
         
        before.append(eval_acc_til)
        before_cil.append(eval_acc_cil)
    
    print("CL Evaluation!")
    '''
    #Read model from saved file
    output_model_file = os.path.join(output_dir, save_name + "_class.pth")
    state_dict = torch.load(output_model_file)
    model.load_state_dict(state_dict)
    '''
    rep = []
    label = []
    for task in range(len(data_all)):
        sentence_val, target_mask_val, label_val, common_mask_eval = data_all[task]['dev']
        local_class = data_all[task]['class'].to(device)
        save_name = data_all[task]['name']
        eval_acc_til, eval_acc_cil, rep_cur, over_smooth, sink_all = evaluate_class(local_class, model, sentence_val, target_mask_val, label_val, batch_size, device, common_mask_eval, type_classifier)
        num_common, sink_dev = sink_all
        #print(num_frequent_all)
        if task == 0:
            over_smooth_eval = over_smooth/total_task_num
            num_common_eval = num_common/total_task_num
            sink_dev_eval = sink_dev/total_task_num
        else:
            over_smooth_eval += over_smooth/total_task_num
            num_common_eval += num_common/total_task_num
            sink_dev_eval += sink_dev/total_task_num
        print("Task "+ str(task), ": The TIL accuracy is:" + str(eval_acc_til) + " The CIL acc is:" + str(eval_acc_cil))
       
        after.append(eval_acc_til)
        after_cil.append(eval_acc_cil)

        #Save some representations for plot
        num_plot = int(277 / 2 * (torch.sum(local_class).cpu().numpy())) 
        rep.extend(rep_cur[:num_plot])
        label.extend(np.array(label_val)[:num_plot, 0].tolist())
    
    if plot_rep:
        #randomly select 1000 samples for visualization
        index = np.random.RandomState(seed=1234).permutation(len(rep))
        vis_index = index[:1000]
        rep_plot = np.array(rep)[vis_index, :].tolist()
        label_plot = np.array(label)[vis_index].tolist()
        visualize(np.array(rep_plot), np.array(label_plot), filename = save_name + '.png')  

    avg_acc = np.mean(after)
    print("average accuracy is:" + str(avg_acc))
    avg_acc_cil = np.mean(after_cil)
    print("average class_incremental accuracy is:" + str(avg_acc_cil))
    if len(before) > 0:
        back_trans = np.sum(np.array(after) - np.array(before))/(len(after) - 1)
        print("Backward transfer is:" + str(back_trans))
        
    ###############################
    #Over-smoothing related results
    ###############################
    print("Over-smoothing (representational similarity) is:")
    for i in range(len(over_smooth_eval)):
        #print("Layer " + str(i))
        print(over_smooth_eval[i])
    #'''
    print("Ratio of sink tokens that are also common tokens is:")
    for i in range(len(num_common_eval)):
        #print("Layer " + str(i))
        print(num_common_eval[i] * 100)

    print("Sink deviation is:")
    for i in range(len(sink_dev_eval)):
        #print("Layer " + str(i))
        print(sink_dev_eval[i])