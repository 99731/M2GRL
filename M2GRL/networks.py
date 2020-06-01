#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import candidate_sampling_ops
from extract_networks import BaseExtractNetwork,TransRExtractNetwork
from config import FLAGS
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import sparse_ops
import tensorflow as tf
import const
import utils

class Network(object):
    def __init__(self, task_id, src_id, dst_id, src_ids, dst_ids, neg_ids, src_type, dst_type, embedding_mgr):
        self.task_id = task_id
        self.src_id = src_id
        self.dst_id = dst_id
        self.src_ids = src_ids
        self.dst_ids = dst_ids
        self.src_type = src_type
        self.dst_type = dst_type
        self.embedding_mgr = embedding_mgr
        self.batch_size = tf.shape(src_id)[0]
        self.positive_size = tf.shape(src_ids)[0]

        self.neg_id = self.negative_sampling()


        self.src_embedding = None
        self.dst_embedding = None

    @utils.func_scope
    def negative_sampling(self):

        func_dict = {
            'log': candidate_sampling_ops.log_uniform_candidate_sampler,
            'learned': candidate_sampling_ops.learned_unigram_candidate_sampler,
            'uniform': candidate_sampling_ops.uniform_candidate_sampler,
            'fix': candidate_sampling_ops.fixed_unigram_candidate_sampler
        }

        num_sampled = const.NODE_CONFIG[self.dst_type]['neg_num'] * FLAGS.batch_size


        parameters = {
            'true_classes': tf.reshape(self.dst_ids, [-1, 1]),
            'num_true': 1,
            'num_sampled': num_sampled,
            'unique': True,
            'range_max': const.NODE_CONFIG[self.dst_type]['max_id'],
            'seed': 123
        }
        if FLAGS.sampling == 'fix':
            parameters.update({
                'vocab_file': FLAGS.vocab_file,
                'distortion': 0.75
            })

        sampled_candidates, true_expected_count, sampled_expected_count = func_dict[FLAGS.sampling](**parameters)

        return tf.reshape(tf.slice(sampled_candidates, [0], [tf.shape(self.dst_id)[0] * const.NODE_CONFIG[self.dst_type]['neg_num']]), [-1, const.NODE_CONFIG[self.dst_type]['neg_num']])

    @utils.func_scope
    def lookup(self):
        src_embedding = self.embedding_mgr.lookup(self.src_id, self.src_type, 'input')
        dst_embedding = self.embedding_mgr.lookup(self.dst_id, self.dst_type, 'output')
        neg_embedding = self.embedding_mgr.lookup(self.neg_id, self.dst_type, 'output')
        src_embeddings = self.embedding_mgr.lookup(self.src_ids, self.src_type, 'input')
        dst_embeddings = self.embedding_mgr.lookup(self.dst_ids, self.dst_type, 'output')
        return src_embedding, dst_embedding, neg_embedding, src_embeddings, dst_embeddings

    @utils.func_scope
    def xent_loss(self, aff, neg_aff):
        if FLAGS.remove_accidental_hits:
            with tf.variable_scope('remove_accidental_hits'):
                neg_ids = tf.reshape(self.neg_id, [-1])
                input_ids = tf.reshape(self.src_id, [-1, 1])
                neg_aff = tf.reshape(neg_aff, [-1])
                acc_hits = candidate_sampling_ops.compute_accidental_hits(input_ids, neg_ids, num_true=1)
                acc_indices, acc_ids, acc_weights = acc_hits
                if neg_aff.dtype != acc_weights.dtype:
                    acc_weights = math_ops.cast(acc_weights, neg_aff.dtype)

                acc_ids = math_ops.cast(acc_ids, dtypes.int32)
                tf.summary.scalar('accidental_hits_num', tf.shape(acc_ids)[0])
                mask_tensor = sparse_ops.sparse_to_dense(acc_ids,
                                                         tf.shape(neg_aff),
                                                         acc_weights,
                                                         default_value=0.0,
                                                         validate_indices=False)
                neg_aff += mask_tensor

        if FLAGS.truncate_affinity:
            with tf.variable_scope('truncate_affinity'):
                embedding_size = tf.cast(const.NODE_CONFIG[self.dst_type]['embedding_size'] / 2.0, tf.float32)
                pos_num = tf.shape(aff)[0]
                aff = tf.gather_nd(aff, tf.where(tf.less(tf.abs(aff), embedding_size)))
                tf.summary.scalar('truncate_pos_affinity', pos_num-tf.shape(aff)[0])

                neg_aff = tf.reshape(neg_aff, [-1])
                neg_num = tf.shape(neg_aff)[0]
                neg_aff = tf.gather_nd(neg_aff, tf.where(tf.less(tf.abs(neg_aff), embedding_size)))

                tf.summary.scalar('truncate_neg_affinity', neg_num-tf.shape(neg_aff)[0])

        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_aff), logits=neg_aff)

        loss = (tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)) / tf.cast(FLAGS.batch_size, tf.float32)

        tf.summary.scalar('loss_'+self.src_type+'2'+self.dst_type, loss)
        return loss

    @utils.func_scope
    def mrr(self, aff, aff_neg):
        aff_all = tf.concat([aff_neg, tf.expand_dims(aff, 1)], 1)
        size = tf.shape(aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
        _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        mrr = tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, -1] + 1)))
        tf.summary.scalar('mrr_'+self.src_type+'2'+self.dst_type, mrr)
        return mrr

    def run(self):
        src_embedding, dst_embedding, neg_embedding, src_embeddings, dst_embeddings = self.lookup()

        if isinstance(self, HeteroNetwork):
            src_embedding = self.extract_network.extract(src_embedding)
            src_embeddings = self.extract_network.extract(src_embeddings)
            dst_embedding = self.extract_network.extract(dst_embedding)
            dst_embeddings = self.extract_network.extract(dst_embeddings)

        self.src_embedding = src_embedding
        self.dst_embedding = dst_embedding
        self.neg_embedding = neg_embedding

        aff = tf.reduce_sum(tf.multiply(src_embedding, dst_embedding), 1)
        neg_aff = utils.neg_cost(src_embedding, neg_embedding)
        pos_aff = tf.reduce_sum(tf.multiply(src_embeddings, dst_embeddings), 1)

        mrr = self.mrr(aff, neg_aff)

        loss = self.xent_loss(pos_aff, neg_aff)

        return loss, mrr

class HomoNetwork(Network):
    def __init__(self, task_id, src_id, dst_id, src_ids, dst_ids, neg_ids, src_type, dst_type, embedding_mgr, ps_num):
        super(HomoNetwork, self).__init__(task_id, src_id, dst_id, src_ids, dst_ids, neg_ids, src_type, dst_type, embedding_mgr)

class HeteroNetwork(Network):
    def __init__(self, task_id, src_id, dst_id, src_ids, dst_ids, neg_ids, src_type, dst_type, embedding_mgr, ps_num):
        super(HeteroNetwork, self).__init__(task_id, src_id, dst_id, src_ids, dst_ids, neg_ids, src_type, dst_type, embedding_mgr)
        if FLAGS.hetero_extract_style == 'base':
            self.extract_network = BaseExtractNetwork(src_type, dst_type, ps_num)
        elif FLAGS.hetero_extract_style == 'transR':
            self.extract_network = TransRExtractNetwork(src_type, dst_type, ps_num)