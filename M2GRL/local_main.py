#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from config import FLAGS
from time import gmtime, strftime
from hetero_graph import HeteroGraph
import tensorflow as tf
import reader
import const

def train():
    last_step = (FLAGS.sample_num / FLAGS.batch_size) * FLAGS.epoch_num
    decay_steps = int(last_step/1.2)
    print('decay_steps', decay_steps)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    learning_rate = tf.constant(FLAGS.learning_rate)
    if FLAGS.lr_decay:
        decay_steps = int(last_step / 200)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, 0.95, True)
        tf.summary.scalar('learning_rate', learning_rate)

    placeholders = reader.local_input_fn()

    model = HeteroGraph(placeholders=placeholders,
                        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                        global_step=global_step,
                        ps_num=1)

    if tf.gfile.Exists('model_dir') and FLAGS.mode == 'train':
        tf.gfile.DeleteRecursively('model_dir')

    tf.set_random_seed(123)

    hooks=[tf.train.StopAtStepHook(last_step=last_step)]

    with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir='model_dir') as mon_sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=mon_sess)
        try:
            for step in range(1000):
                if FLAGS.mode == 'train':
                    _, g = mon_sess.run([model.opt, model.global_step])

                    # i2c_loss, i2c_mrr, i2c_bs, i2c_ps, \
                    # c2c_loss, c2c_mrr, c2c_bs, c2c_ps, \
                    if step % 100 == 0:
                        _, g, total_loss, \
                        i2i_loss, i2i_mrr, i2i_bs, i2i_ps, \
                        lr, neg_id = mon_sess.run(
                            [model.opt,
                             global_step,
                             model.loss,
                             model.loss_dict['item2item'],
                             model.mrr_dict['item2item'],
                             model.network_dict['item2item'].batch_size,
                             model.network_dict['item2item'].positive_size,
                             learning_rate,
                             model.network_dict['item2item'].neg_id,
                             ])
                        print('#' * 70)
                        print('[{}]step:{}, total_loss={:.3f}, lr={:.8f}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), g, total_loss, lr))
                        print('[item2item] loss={:.3f}, mrr={:.3f}, batch_size={}, positive_size={}'.format(i2i_loss, i2i_mrr, i2i_bs, i2i_ps))
                        print('neg_id', neg_id.tolist())

                elif FLAGS.mode == 'export':
                    g, src_item_ids, src_embeddings, dst_embeddings = mon_sess.run([global_step, model.network_dict['item2item'].src_id, model.network_dict['item2item'].src_embedding, model.network_dict['item2item'].dst_embedding])
                    src_item_ids = src_item_ids.tolist()
                    src_embeddings = src_embeddings.tolist()
                    dst_embeddings = dst_embeddings.tolist()

                    records = []
                    for src_item_id, src_embedding, dst_embedding in zip(src_item_ids, src_embeddings, dst_embeddings):
                        records.append((str(src_item_id), ','.join(map(str, src_embedding)), ','.join(map(str, dst_embedding))))
                    if step % 100 == 0:
                        print('global_step=[{}]. step=[{}], item_id=[{}], src_embedding=[{}], dst_embedding=[{}]'.format(g, step, src_item_ids[0], src_embeddings[0][0:5], dst_embeddings[0][0:5]))


        except tf.errors.OutOfRangeError:
            print('training for 1 epochs, %d steps'%step)
        finally:
            coord.request_stop()
            coord.join(threads)

def main(argv=None):
    # if FLAGS.mode == 'train':
    train()


if __name__ == '__main__':
    tf.app.run()