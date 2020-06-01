#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from config import FLAGS
import utils, const
import math

class EmbeddingMgr(object):
    @utils.class_scope
    def __init__(self, ps_num):
        self.embeddings = {}
        self.ps_num = ps_num

        for node_type, config in const.NODE_CONFIG.iteritems():
            max_id = config['max_id']
            embedding_size = config['embedding_size']
            init_width = 0.5 / embedding_size
            if FLAGS.tying:
                with tf.variable_scope(name_or_scope=node_type, partitioner=utils.get_partition(self.ps_num)):
                    self.embeddings[node_type] = tf.get_variable(name=node_type+'_embedding',
                                                                 shape=[max_id, embedding_size],
                                                                 initializer=tf.initializers.random_uniform(-init_width, init_width),)
            else:
                with tf.variable_scope(name_or_scope=node_type+'_input', partitioner=utils.get_partition(self.ps_num)):
                    self.embeddings[node_type + '_input'] = tf.get_variable(node_type+'_embedding_input', [max_id, embedding_size],
                                                                            initializer=tf.initializers.random_uniform(-init_width, init_width))

                if node_type != 'user':
                    with tf.variable_scope(name_or_scope=node_type+'_output', partitioner=utils.get_partition(self.ps_num)):
                        self.embeddings[node_type + '_output'] = tf.get_variable(node_type+'_embedding_output', [max_id, embedding_size],
                                                                                 initializer=tf.initializers.zeros())

    def lookup(self, id_feature, node_type, type):
        if FLAGS.tying:
            return tf.nn.embedding_lookup(self.embeddings[node_type], id_feature)
        else:
            return tf.nn.embedding_lookup(self.embeddings[node_type+'_'+type], id_feature)