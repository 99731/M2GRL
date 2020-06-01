#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from config import FLAGS

def xent_loss(aff, neg_aff):
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(aff), logits=aff)
    negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(neg_aff), logits=neg_aff)

    loss = (tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)) / tf.cast(FLAGS.batch_size, tf.float32)
    return loss