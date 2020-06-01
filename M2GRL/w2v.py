#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.platform import tf_logging
from tensorflow.python.ops import candidate_sampling_ops
from const import *
from extract_networks import BaseExtractNetwork
from config import FLAGS
from utils import *
from embedding_mgr import EmbeddingMgr
from networks import *
import tensorflow as tf

MAX_ID_CONFIG = {
    'item': const.NODE_CONFIG['item']['max_id'],
    'category': const.NODE_CONFIG['category']['max_id']
}

class W2V(object):
    @class_scope
    def __init__(self,
                 placeholders,
                 optimizer,
                 global_step,
                 ps_num):

        self.global_step = global_step
        self.optimizer = optimizer
        self.placeholders = placeholders
        self.ps_num = ps_num
        self.build()

    def build(self):
        def _negative_sampling(dst_id, sample_method=FLAGS.sample):
            sample_func = {'learned':candidate_sampling_ops.learned_unigram_candidate_sampler,
                           'log':candidate_sampling_ops.log_uniform_candidate_sampler,
                           'uniform':candidate_sampling_ops.uniform_candidate_sampler}
            sampled_candidates, true_expected_count, sampled_expected_count = sample_func[sample_method](
                true_classes=tf.reshape(dst_id, [-1, 1]),
                num_true=1,
                num_sampled=FLAGS.neg_num*FLAGS.batch_size,
                unique=True,
                range_max=MAX_ID_CONFIG['item'],
                seed=123)
            return sampled_candidates, true_expected_count, sampled_expected_count

        src_id = self.placeholders['src_id']
        dst_id = self.placeholders['dst_id']
        sampled_candidates, true_expected_count, sampled_expected_count = _negative_sampling(dst_id)

        # mrr calculate
        neg_id, _, _ = _negative_sampling(dst_id, 'uniform')
        neg_id = tf.reshape(neg_id, [FLAGS.batch_size, FLAGS.neg_num])

        with tf.variable_scope('embeddings', partitioner=get_partition(self.ps_num)):
            item_embeddings = tf.get_variable('item_embeddings', [MAX_ID_CONFIG['item'], FLAGS.embedding_size])
            if not FLAGS.tying:
                nce_weights = tf.get_variable('nce_weights', [MAX_ID_CONFIG['item'], FLAGS.embedding_size])
            else:
                nce_weights = item_embeddings
            nce_biases = tf.get_variable('nce_biases', [MAX_ID_CONFIG['item']], initializer=tf.zeros_initializer())

        # embedding lookup
        src_embedding = tf.nn.embedding_lookup(item_embeddings, src_id)

        if FLAGS.batch_norm:
            src_embedding = tf.layers.batch_normalization(src_embedding, training=(FLAGS.mode=="train"))

        dst_embedding = tf.nn.embedding_lookup(nce_weights, dst_id)
        neg_embedding = tf.nn.embedding_lookup(nce_weights, neg_id)

        # calculate mrr
        aff = affinity(src_embedding, dst_embedding)
        neg_aff = neg_cost(src_embedding, neg_embedding)
        mrr = self.mrr(aff, neg_aff)

        loss_func = {'nce':tf.nn.nce_loss, 'softmax':tf.nn.sampled_softmax_loss}

        # sampled softmax loss
        loss = loss_func[FLAGS.xent_loss](weights=nce_weights,
                                          biases=nce_biases,
                                          labels=tf.reshape(dst_id, [-1, 1]),
                                          inputs=tf.nn.embedding_lookup(item_embeddings, src_id),
                                          num_sampled=FLAGS.neg_num*FLAGS.batch_size,
                                          num_classes=MAX_ID_CONFIG['item'],
                                          remove_accidental_hits=True,
                                          sampled_values=(sampled_candidates, true_expected_count, sampled_expected_count))

        self.loss = tf.reduce_mean(loss)

        tf.summary.scalar('loss', self.loss)
        self.opt = self.optimizer.minimize(self.loss, self.global_step)

        self.loss_dict = {'item2item':self.loss,}
        self.mrr_dict = {'item2item':mrr}
        self.src_id = src_id
        self.dst_id = dst_id
        self.neg_id = neg_id
        self.src_embedding = src_embedding
        self.dst_embedding = dst_embedding
        self.neg_embedding = neg_embedding

    def mrr(self, aff, aff_neg):
        aff_all = tf.concat([aff_neg, tf.expand_dims(aff, 1)], axis=1)
        size = tf.shape(aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
        _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        mrr = tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, -1] + 1)))
        tf.summary.scalar('mrr_item2item', mrr)
        return mrr

def affinity(inputs1, inputs2):
    return tf.reduce_sum(tf.multiply(inputs1, inputs2), axis=1)

def neg_cost(inputs1, negatives):
    inputs1 = tf.expand_dims(inputs1, 1)
    result = tf.multiply(inputs1, negatives)
    result = tf.reduce_sum(result, -1)
    return result