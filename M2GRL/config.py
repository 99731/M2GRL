#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('window_size', 9, '')
flags.DEFINE_float('learning_rate', 0.01, '')
flags.DEFINE_float('weight_decay', 0.0, '')
flags.DEFINE_integer('epoch_num', 1000, '')
flags.DEFINE_boolean('lr_decay', True, '')
flags.DEFINE_string('sampling', 'uniform', '')
flags.DEFINE_string('loss', 'xent', '')
flags.DEFINE_boolean('truncate_affinity', False, '')
flags.DEFINE_boolean('remove_accidental_hits', False, '')
flags.DEFINE_boolean('tying', False, '')
flags.DEFINE_integer('export_type', 2, '')

flags.DEFINE_string("volumes", "volumes", "volumes info")
flags.DEFINE_string('tables', '', 'odps tables')
flags.DEFINE_integer('task_index', None, 'Worker task index')
flags.DEFINE_string('ps_hosts', '','ps hosts')
flags.DEFINE_string('worker_hosts', '', 'worker hosts')
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
flags.DEFINE_string('checkpoint_path', "model_dir", 'job name: worker or ps')
flags.DEFINE_string("checkpointDir", "", "checkpoint_dir")
flags.DEFINE_string("mode", "train", "")
flags.DEFINE_integer('sample_num', 100, '')
flags.DEFINE_string('exportModelDir', '', '')
flags.DEFINE_string('output_embedding_table', '', '')
flags.DEFINE_string('vocab_file', 'input/vocab', '')
flags.DEFINE_float('distortion', 0.75, '')
flags.DEFINE_boolean('tensor_fuse', True, '')
flags.DEFINE_string('hetero_extract_style', 'transR', 'how to project feature vectors in hetero, support values: base, transR')