#coding=utf-8
from config import FLAGS
import tensorflow as tf
import utils
import const

class BaseExtractNetwork(object):
    @utils.class_scope
    def __init__(self, src_type, dst_type, ps_num):
        self.src_type = src_type
        self.dst_type = dst_type
        with tf.variable_scope(name_or_scope='W_'+src_type+'2'+dst_type, partitioner=utils.get_partition(ps_num)):
            self.W = tf.get_variable('W_'+src_type+'2'+dst_type, [const.NODE_CONFIG[self.src_type]['embedding_size'], const.NODE_CONFIG[self.dst_type]['embedding_size']])# FLAGS.embedding_size, FLAGS.embedding_size])

        with tf.variable_scope(name_or_scope='b_'+src_type+'2'+dst_type, partitioner=utils.get_partition(ps_num)):
            self.b = tf.get_variable('b_'+src_type+'2'+dst_type, [const.NODE_CONFIG[self.dst_type]['embedding_size'],])

    @utils.func_scope
    def extract(self, src):
        return tf.sigmoid(tf.matmul(src, self.W) + self.b)


class TransRExtractNetwork(object):
    @utils.class_scope
    def __init__(self, src_type, dst_type, ps_num):
        self.src_type = src_type
        self.dst_type = dst_type
        self.transR_embedding_size = const.TRANS_DICT['transR_embedding_size']

        # check if the embedding sizes of two type are equal
        assert const.NODE_CONFIG[self.src_type]['embedding_size'] == const.NODE_CONFIG[self.dst_type]['embedding_size']

        with tf.variable_scope(name_or_scope='W_'+src_type+'2'+dst_type, partitioner=utils.get_partition(ps_num)):
            self.W = tf.get_variable('W_'+src_type+'2'+dst_type, [const.NODE_CONFIG[self.src_type]['embedding_size'], self.transR_embedding_size])# FLAGS.embedding_size, FLAGS.embedding_size])

    @utils.func_scope
    def extract(self, embed):
        return tf.sigmoid(tf.matmul(embed, self.W))