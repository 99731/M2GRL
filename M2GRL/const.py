#coding=utf-8

NODE_CONFIG = {
    'item': {
        'max_id': 1000, # vocabulary size
        'neg_num': 10, # num of negative samples
        'embedding_size': 64,
        'sampling': 'uniform' # method of negative sampling
    },
    'category': {
        'max_id': 120,
        'neg_num': 2,
        'embedding_size': 64,
        'sampling': 'uniform'
    }
}

TASK_CONFIG = {
    0:('item', 'category'), # inter-task 'item2category'
    1:('item', 'item'), # intra-task 'item2item'
    2:('category', 'category') # intra-task 'category2category'
}

ALPHA_DICT = {
    'item2category':1,
    'item2item':1,
    'category2category':1
}

TRANS_DICT = {
    'transR_embedding_size' : 64,
}