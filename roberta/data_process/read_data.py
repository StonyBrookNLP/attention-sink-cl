from data_process.data_all import data_rte, data_mrpc, data_sst, data_qqp, data_mnli, data_ag, data_db, data_yahoo, data_yelp, data_amazon

#Extract the data for each task given the sequence name and the subset name
def read_data(dataset, subset = None, args = None, word_to_global_label = {}, ratio_train = 1):
    bsz = args.bsize
    if 'yahoo' in dataset or subset == 'yahoo': 
        if subset is not None and subset != 'yahoo':
            n_class = 2
            if '10000' in dataset:
                taskpath = f'../../data/Yahoo10000_{subset}'
            else:
                taskpath = f'../../data/Yahoo_{subset}'
        else:
            n_class = 10 
            taskpath = '../../data/Yahoo'
        data, word_to_global_label = data_yahoo(taskpath, ratio_train, True, word_to_global_label, sublabel = subset) 
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        max_len = 128
    elif dataset == 'amazon' or subset == 'amazon': 
        n_class = 5
        taskpath = '../../data/Amazon'
        data, word_to_global_label = data_amazon(taskpath, ratio_train, True, word_to_global_label) 
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        max_len = 128
    elif dataset == 'yelp' or subset == 'yelp':
        n_class = 5
        taskpath = '../../data/Yelp'
        data, word_to_global_label = data_yelp(taskpath, ratio_train, True, word_to_global_label) 
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        max_len = 128
    elif dataset == 'ag_news' or subset == 'ag_news': 
        n_class = 4 
        taskpath = '../../data/AG_News'
        data, word_to_global_label = data_ag(taskpath, ratio_train, True, word_to_global_label, sublabel = subset) 
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        max_len = 128 
    elif dataset == 'ag_split':
        n_class = 2
        taskpath = f'../../data/AG_News_{subset}'
        data, word_to_global_label = data_ag(taskpath, ratio_train, True, word_to_global_label, sublabel = subset) 
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        max_len = 128   
    elif 'db' in dataset or subset == 'db':
        if subset is not None and subset != 'db':
            n_class = 2
            taskpath = f'../../data/DBPedia_{subset}' 
        else:
            taskpath = '../../data/DBPedia'
            n_class = 14 
        data, word_to_global_label = data_db(taskpath, ratio_train, True, word_to_global_label, sublabel = subset) 
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        max_len = 128
    elif dataset == 'rte' or subset == 'rte':
        taskpath = '../../data/RTE'
        data, word_to_global_label = data_rte(taskpath, ratio_train, True, word_to_global_label)  
        batch_size = bsz if 'base' in args.enc_type else 8
        n_class = 2
        max_len = 512
    elif dataset == 'mnli-m' or subset == 'mnli-m':
        taskpath = '../../data/MNLI-m'
        data, word_to_global_label = data_mnli(taskpath, ratio_train, True, word_to_global_label) 
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        n_class = 3
        max_len = 128
    elif dataset == 'mrpc' or subset == 'mrpc':
        taskpath = '../../data/MRPC'
        data, word_to_global_label = data_mrpc(taskpath, ratio_train, True, word_to_global_label) 
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        n_class = 2
        max_len = 128
    elif dataset == 'qnli' or subset == 'qnli':
        taskpath = '../../data/QNLI'
        data, word_to_global_label = data_rte(taskpath, ratio_train, True, word_to_global_label)
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        n_class = 2
        max_len = 128
    elif dataset == 'sst' or subset == 'sst':
        taskpath = '../../data/SST'
        data, word_to_global_label = data_sst(taskpath, ratio_train, True, word_to_global_label)
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        n_class = 2
        max_len = 128
    elif dataset == 'qqp' or subset == 'qqp':
        taskpath = '../../data/QQP'
        data, word_to_global_label = data_qqp(taskpath, ratio_train, True, word_to_global_label) 
        batch_size = bsz if 'base' in args.enc_type else bsz // 2
        n_class = 2
        max_len = 128
    return data, batch_size, n_class, max_len, word_to_global_label