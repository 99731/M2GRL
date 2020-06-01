

## M2GRL

This is a demo of the M2GRL framework, which is designed for learning node representations from multi-view graphs for web-scale recommender Recommender systems.

For more details, please refer to our KDD2020 paper ["M2GRL: A Multi-task Multi-view Graph Representation Learning Framework for Web-scale Recommender Systems"](https://arxiv.org/abs/2005.10110)

Contact: Menghan Wang (wangmengh@zju.edu.cn)

## Requirements

- python version: 2.7

- tensorflow verison: 1.8


## Data format

A typical example of data has six columns: 'src_ids', 'dst_ids', 'src_id', 'dst_id', 'type', 'neg_ids'

'type': an integer indicates which task this data sample belongs to. The mapping configuration is defined in `const.py` (the key of dict `TASK_CONFIG`).
'src_id': the first element of data pair ('src_id','dst_id').
'dst_id': the second element of data pair ('src_id','dst_id').
'neg_ids': the sampled negative ids.
'src_ids': a fixed-length sequence of 'src_id', designed for efficient training. 
'dst_ids': a fixed-length sequence of 'dst_id', designed for efficient training.

Note that in practice we aggregate data pairs by src_id and generate 'src_ids' and 'dst_ids'.

Below is an example in `reader.py`.
```
features={
            'src_ids': tf.FixedLenFeature([FLAGS.window_size*2], tf.int64),
            'dst_ids': tf.FixedLenFeature([FLAGS.window_size*2], tf.int64),
            'src_id': tf.FixedLenFeature([], tf.int64),
            'dst_id': tf.FixedLenFeature([], tf.int64),
            'type': tf.FixedLenFeature([], tf.int64),
            'neg_ids': tf.FixedLenFeature([20], tf.int64),
        })
```

## How to set tasks
Configurations of tasks and views are in the `const.py`, where a sample config is given. 


## How to run
1. run command `python reader.py` to generate example data.

2. run command `python local_main.py` to run the model.
Set `FLAGS.mode = "train"` to train the model and `FLAGS.mode = "export"` to get the learned representations.
 

## Citation
Please cite our paper if it is helpful to your research:
```
@article{DBLP:journals/corr/abs-2005-10110,
  author    = {Menghan Wang and
               Yujie Lin and
               Guli Lin and
               Keping Yang and
               Xiao{-}Ming Wu},
  title     = {{M2GRL:} {A} Multi-task Multi-view Graph Representation Learning Framework
               for Web-scale Recommender Systems},
  journal   = {CoRR},
  volume    = {abs/2005.10110},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.10110},
  archivePrefix = {arXiv},
  eprint    = {2005.10110},
  timestamp = {Fri, 22 May 2020 16:21:28 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2005-10110.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
