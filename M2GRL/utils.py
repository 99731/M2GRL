#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import partitioned_variables

import tensorflow as tf
import numpy as np
from functools import wraps

def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def affinity(inputs1, inputs2):
    return tf.reduce_sum(tf.multiply(inputs1, inputs2), axis=1)

def neg_cost(inputs1, negatives):
    inputs1 = tf.expand_dims(inputs1, 1)
    result = tf.multiply(inputs1, negatives)
    result = tf.reduce_sum(result, -1)
    return result

def func_scope(func):
    @wraps(func)
    def _wrapper(*arg, **kwargs):
        with tf.variable_scope(func.__name__):
            return func(*arg, **kwargs)
    return _wrapper

def class_scope(func):
    @wraps(func)
    def _wrapper(*arg, **kwargs):
        with tf.variable_scope(arg[0].__class__.__name__):
            return func(*arg, **kwargs)
    return _wrapper

def get_partition(max_partitions=None, min_slice_size=32):

    partition_id = partitioned_variables.min_max_variable_partitioner(
        max_partitions=max_partitions,
        min_slice_size=min_slice_size << 20)

    return partition_id