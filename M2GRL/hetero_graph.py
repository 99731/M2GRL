#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from embedding_mgr import EmbeddingMgr
from networks import HeteroNetwork, HomoNetwork
from multi_loss_handler import MultiLossLayer
import tensorflow as tf
import const
import utils

class HeteroGraph(object):
    @utils.class_scope
    def __init__(self,
                 placeholders,
                 optimizer,
                 global_step,
                 ps_num):
        self.loss = 0.0
        self.ps_num = ps_num
        self.loss_dict = {}
        # self.loss_list = []
        self.mrr_dict = {}
        self.network_dict = {}
        self.global_step = global_step
        self.optimizer = optimizer
        self.placeholders = placeholders
        self.input_mgr = InputMgr(placeholders)
        self.embedding_mgr = EmbeddingMgr(ps_num)
        self.multi_loss_handler = MultiLossLayer(const.TASK_CONFIG)
        self.build_networks()
        self.opt = self.opt()


    def build_networks(self):
        for type_id, (src_type, dst_type) in const.TASK_CONFIG.iteritems():
            name = src_type+'2'+dst_type
            if src_type == dst_type:
                with tf.variable_scope('HomoNetwork_'+src_type+'2'+dst_type):
                    self.network_dict[name] = HomoNetwork(type_id,
                                                          self.placeholders['src_' + name + '_id'],
                                                          self.placeholders['dst_' + name + '_id'],
                                                          self.placeholders['src_' + name + '_ids'],
                                                          self.placeholders['dst_' + name + '_ids'],
                                                          self.placeholders['neg_' + name + '_ids'],
                                                          src_type,
                                                          dst_type,
                                                          self.embedding_mgr,
                                                          self.ps_num)
            else:
                with tf.variable_scope('HeteroNetwork_'+src_type+'2'+dst_type):
                    self.network_dict[name] = HeteroNetwork(type_id,
                                                            self.placeholders['src_' + name + '_id'],
                                                            self.placeholders['dst_' + name + '_id'],
                                                            self.placeholders['src_' + name + '_ids'],
                                                            self.placeholders['dst_' + name + '_ids'],
                                                            self.placeholders['neg_' + name + '_ids'],
                                                            src_type,
                                                            dst_type,
                                                            self.embedding_mgr,
                                                            self.ps_num)
            # dynamic weights
            with tf.variable_scope('run_'+name):
                loss, mrr = self.network_dict[name].run()
                self.loss_dict[name] = tf.cond(tf.is_nan(loss), lambda:0.0, lambda:loss)  #* const.ALPHA_DICT.get(name, 1.0)
                #self.loss_list.append(tf.cond(tf.is_nan(loss), lambda:0.0, lambda:loss))
                self.mrr_dict[name] = mrr

            ''' static weights
            with tf.variable_scope('run_'+name):
                loss, mrr = self.network_dict[name].run()
                self.loss_dict[name] = loss * const.ALPHA_DICT.get(name, 1.0)
                self.mrr_dict[name] = mrr
            '''
        self.loss = self.multi_loss_handler.get_loss(list(self.loss_dict.values()))


    def opt(self):
        gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
        clipped_gradients = []
        for g in gradients:
            if isinstance(g, tf.Tensor):
                clipped_gradients.append(tf.clip_by_value(g, -1.0, 1.0))
            elif isinstance(g, tf.IndexedSlices):
                clipped_gradients.append(tf.IndexedSlices(tf.clip_by_value(g.values, -1.0, 1.0), g.indices, g.dense_shape))
            else:
                clipped_gradients.append(g)
        return self.optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=self.global_step)

class InputMgr(object):
    @utils.class_scope
    def __init__(self, placeholders):
        src_id = placeholders['src_id']
        dst_id = placeholders['dst_id']
        src_ids = placeholders['src_ids']
        dst_ids = placeholders['dst_ids']
        type = placeholders['type']
        neg_ids = placeholders['neg_ids']

        for type_id, (src_type, dst_type) in const.TASK_CONFIG.iteritems():
            _src_id = tf.gather_nd(src_id, tf.where(tf.equal(type, type_id)))
            _dst_id = tf.gather_nd(dst_id, tf.where(tf.equal(type, type_id)))
            _src_ids = tf.gather_nd(src_ids, tf.where(tf.equal(type, type_id)))
            _dst_ids = tf.gather_nd(dst_ids, tf.where(tf.equal(type, type_id)))
            _neg_ids = tf.gather_nd(neg_ids, tf.where(tf.equal(type, type_id)))

            print('_neg_ids', _neg_ids)

            _flat_src_ids = tf.reshape(_src_ids, [-1])
            _flat_dst_ids = tf.reshape(_dst_ids, [-1])

            placeholders['src_' + src_type + '2' + dst_type + '_id'] = _src_id
            placeholders['dst_' + src_type + '2' + dst_type + '_id'] = _dst_id
            placeholders['neg_'+src_type+'2'+dst_type+'_ids'] = _neg_ids
            placeholders['src_'+src_type+'2'+dst_type+'_ids'] = tf.gather_nd(_flat_src_ids, tf.where(tf.greater(_flat_dst_ids, 0)))
            placeholders['dst_'+src_type+'2'+dst_type+'_ids'] = tf.gather_nd(_flat_dst_ids, tf.where(tf.greater(_flat_dst_ids, 0)))