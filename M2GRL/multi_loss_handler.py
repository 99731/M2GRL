#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim

class MultiLossLayer():
    def __init__(self, loss_name_dict):
        self.threshold = tf.constant(value=0.05, dtype=tf.float32)
        self._sigmas_sq = []
        for type_id, (src_type, dst_type) in loss_name_dict.iteritems():
            name = src_type + '2' + dst_type
            self._sigmas_sq.append(slim.variable('Sigma_sq_' + name, dtype=tf.float32, shape=[],
                                                 initializer=tf.initializers.random_uniform(minval=0.2, maxval=1)))
            tf.summary.scalar('Sigma_sq_' + name,self._sigmas_sq[-1])

    def get_loss(self, loss_list):
        self._sigmas_sq[0] = tf.cond(tf.less(self._sigmas_sq[0], self.threshold), lambda: self.threshold, lambda: self._sigmas_sq[0])
        factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[0]))

        loss = tf.add(tf.multiply(factor,  loss_list[0]), tf.log(self._sigmas_sq[0]))
        for i in range(1, len(self._sigmas_sq)):
            self._sigmas_sq[i] = tf.cond(tf.less(self._sigmas_sq[i], self.threshold), lambda: self.threshold, lambda: self._sigmas_sq[i])
            factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[i]))
            loss = tf.add(loss, tf.add(tf.multiply(factor, loss_list[i]), tf.log(self._sigmas_sq[i])))

        tf.summary.scalar('total_loss', loss)
        return loss
