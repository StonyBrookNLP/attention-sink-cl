import numpy as np

# Extract name of each task given the name of a sequence, e.g. news_series.
def get_subset(dataset):
    if dataset == 'yahoo_split':
        subset = ['1', '2', '3', '4', '5']
    elif dataset == 'yahoo10000_split':
        subset = ['1', '2', '3', '4', '5']
    elif dataset == 'ag_split':
        subset = ['1', '2']
    elif dataset == 'db_split':
        subset = ['1', '2', '3', '4', '5', '6', '7']
    elif dataset == 'news_series':
        subset = ['ag_news', 'sst', 'rte', 'mrpc']
    elif dataset == 'news_series_mnli':
        subset = ['ag_news', 'sst', 'rte', 'mrpc', 'mnli']
    elif dataset == 'review':
        subset = ['ag_news', 'yelp', 'amazon', 'db', 'yahoo']
    elif dataset == 'glue':
        subset = ['sst', 'qnli', 'mnli-m', 'mrpc', 'amazon', 'ag_news', 'qqp']
    else:
        subset = [None]
        
    if subset[0] is not None:
        subset_index = np.random.permutation(len(subset))
        subset = np.array(subset, dtype = object)[subset_index].tolist()
    return subset