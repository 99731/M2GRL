#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import utils
from config import FLAGS
import const
import random

@utils.func_scope
def local_input_fn():
    return _local_input_fn(['input/tfrecords.file'])

def _local_input_fn(file_list):

    min_after_dequeue = FLAGS.batch_size * 10
    capacity = min_after_dequeue + 3 * FLAGS.batch_size

    file_queue = tf.train.string_input_producer(file_list, num_epochs=FLAGS.epoch_num, shuffle=False, capacity=capacity)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'src_ids': tf.FixedLenFeature([FLAGS.window_size*2], tf.int64),
            'dst_ids': tf.FixedLenFeature([FLAGS.window_size*2], tf.int64),
            'src_id': tf.FixedLenFeature([], tf.int64),
            'dst_id': tf.FixedLenFeature([], tf.int64),
            'type': tf.FixedLenFeature([], tf.int64),
            'neg_ids': tf.FixedLenFeature([20], tf.int64),
        })

    features = tf.train.batch(features,
                              batch_size=FLAGS.batch_size,
                              capacity=100 + FLAGS.batch_size)

    return features

def example_generator(item_num, cate_num, type, item_cate_map):
    # types:
    # 0:('item', 'category'),  # inter-task
    # 1:('item', 'item'),  # intra-task
    # 2:('category', 'category')  # intra-task
    if type == 0:
        src_id = random.randint(0, item_num-1)
        if src_id in item_cate_map:
            dst_id = item_cate_map[src_id]
        else:
            dst_id = random.randint(0, cate_num-1)
            item_cate_map.update({src_id:dst_id})
    elif type == 1:
        src_id = random.randint(0, item_num-1)
        dst_id = random.randint(0, item_num-1)
    elif type == 2:
        src_id = random.randint(0, cate_num-1)
        dst_id = random.randint(0, cate_num-1)

    neg_range = item_num if type ==2 else cate_num
    return src_id, dst_id, neg_range


if __name__ == '__main__':

    item_num = const.NODE_CONFIG['item']['max_id']
    cate_num = const.NODE_CONFIG['category']['max_id']
    sample_num = 1000
    print('item_num:',item_num,'cate_num:',cate_num)


    writer = tf.python_io.TFRecordWriter('input/tfrecords.file')


    mapping_dict = {}
    index = 0
    src_id, dst_id = 0, 0
    item_cate_map = {}
    for line in range(sample_num):
        type = random.choice([0, 1, 2])
        src_id,dst_id, neg_range = example_generator(item_num, cate_num, type, item_cate_map)



        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'src_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=[src_id for i in range(FLAGS.window_size*2)])),
                    'dst_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=[dst_id for i in range(FLAGS.window_size*2)])),
                    'src_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[src_id])),
                    'dst_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[dst_id])),
                    'type': tf.train.Feature(int64_list=tf.train.Int64List(value=[type])),
                    'neg_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=[random.randint(0, neg_range-1) for i in range(20)])),
                }
            )
        )
        writer.write(example.SerializeToString())
