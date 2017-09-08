# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert checkpoints using RNNCells to new name convention.

Usage:

  python checkpoint_convert.py [--write_v1_checkpoint] \
      '/path/to/checkpoint' '/path/to/new_checkpoint'

For example, if there is a V2 checkpoint to be converted and the files include:
  /tmp/my_checkpoint/model.ckpt.data-00000-of-00001
  /tmp/my_checkpoint/model.ckpt.index
  /tmp/my_checkpoint/model.ckpt.meta

use the following command:
  mkdir /tmp/my_converted_checkpoint &&
  python checkpoint_convert.py \
      /tmp/my_checkpoint/model.ckpt /tmp/my_converted_checkpoint/model.ckpt

This will generate three converted checkpoint files corresponding to the three
old ones in the new directory:
  /tmp/my_converted_checkpoint/model.ckpt.data-00000-of-00001
  /tmp/my_converted_checkpoint/model.ckpt.index
  /tmp/my_converted_checkpoint/model.ckpt.meta
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import re
import sys

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import app
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib

# Mapping between old <=> new names. Externalized so that user scripts that
# may need to consume multiple checkpoint formats can use this metadata.
RNN_NAME_REPLACEMENTS = collections.OrderedDict([
    ############################################################################
    # contrib/rnn/python/ops/core_rnn_cell_impl.py
    # BasicRNNCell
    ('basic_rnn_cell/weights', 'basic_rnn_cell/kernel'),
    ('basic_rnn_cell/biases', 'basic_rnn_cell/bias'),
    # GRUCell
    ('gru_cell/weights', 'gru_cell/kernel'),
    ('gru_cell/biases', 'gru_cell/bias'),
    ('gru_cell/gates/weights', 'gru_cell/gates/kernel'),
    ('gru_cell/gates/biases', 'gru_cell/gates/bias'),
    ('gru_cell/candidate/weights', 'gru_cell/candidate/kernel'),
    ('gru_cell/candidate/biases', 'gru_cell/candidate/bias'),
    # BasicLSTMCell
    ('basic_lstm_cell/weights', 'basic_lstm_cell/kernel'),
    ('basic_lstm_cell/biases', 'basic_lstm_cell/bias'),
    # LSTMCell
    ('lstm_cell/weights', 'lstm_cell/kernel'),
    ('lstm_cell/biases', 'lstm_cell/bias'),
    ('lstm_cell/projection/weights', 'lstm_cell/projection/kernel'),
    ('lstm_cell/projection/biases', 'lstm_cell/projection/bias'),
    # OutputProjectionWrapper
    ('output_projection_wrapper/weights', 'output_projection_wrapper/kernel'),
    ('output_projection_wrapper/biases', 'output_projection_wrapper/bias'),
    # InputProjectionWrapper
    ('input_projection_wrapper/weights', 'input_projection_wrapper/kernel'),
    ('input_projection_wrapper/biases', 'input_projection_wrapper/bias'),
    ############################################################################
    # contrib/rnn/python/ops/lstm_ops.py
    # LSTMBlockFusedCell ??
    ('lstm_block_wrapper/weights', 'lstm_block_wrapper/kernel'),
    ('lstm_block_wrapper/biases', 'lstm_block_wrapper/bias'),
    ############################################################################
    # contrib/rnn/python/ops/rnn_cell.py
    # LayerNormBasicLSTMCell
    ('layer_norm_basic_lstm_cell/weights', 'layer_norm_basic_lstm_cell/kernel'),
    ('layer_norm_basic_lstm_cell/biases', 'layer_norm_basic_lstm_cell/bias'),
    # UGRNNCell, not found in g3, but still need it?
    ('ugrnn_cell/weights', 'ugrnn_cell/kernel'),
    ('ugrnn_cell/biases', 'ugrnn_cell/bias'),
    # NASCell
    ('nas_rnn/weights', 'nas_rnn/kernel'),
    ('nas_rnn/recurrent_weights', 'nas_rnn/recurrent_kernel'),
    # IntersectionRNNCell
    ('intersection_rnn_cell/weights', 'intersection_rnn_cell/kernel'),
    ('intersection_rnn_cell/biases', 'intersection_rnn_cell/bias'),
    ('intersection_rnn_cell/in_projection/weights',
     'intersection_rnn_cell/in_projection/kernel'),
    ('intersection_rnn_cell/in_projection/biases',
     'intersection_rnn_cell/in_projection/bias'),
    # PhasedLSTMCell
    ('phased_lstm_cell/mask_gates/weights',
     'phased_lstm_cell/mask_gates/kernel'),
    ('phased_lstm_cell/mask_gates/biases', 'phased_lstm_cell/mask_gates/bias'),
    ('phased_lstm_cell/new_input/weights', 'phased_lstm_cell/new_input/kernel'),
    ('phased_lstm_cell/new_input/biases', 'phased_lstm_cell/new_input/bias'),
    ('phased_lstm_cell/output_gate/weights',
     'phased_lstm_cell/output_gate/kernel'),
    ('phased_lstm_cell/output_gate/biases',
     'phased_lstm_cell/output_gate/bias'),
    # AttentionCellWrapper
    ('attention_cell_wrapper/weights', 'attention_cell_wrapper/kernel'),
    ('attention_cell_wrapper/biases', 'attention_cell_wrapper/bias'),
    ('attention_cell_wrapper/attn_output_projection/weights',
     'attention_cell_wrapper/attn_output_projection/kernel'),
    ('attention_cell_wrapper/attn_output_projection/biases',
     'attention_cell_wrapper/attn_output_projection/bias'),
    ('attention_cell_wrapper/attention/weights',
     'attention_cell_wrapper/attention/kernel'),
    ('attention_cell_wrapper/attention/biases',
     'attention_cell_wrapper/attention/bias'),
    ############################################################################
    # contrib/legacy_seq2seq/python/ops/seq2seq.py
    ('attention_decoder/weights',
     'attention_decoder/kernel'),
    ('attention_decoder/biases',
     'attention_decoder/bias'),
    ('attention_decoder/Attention_0/weights',
     'attention_decoder/Attention_0/kernel'),
    ('attention_decoder/Attention_0/biases',
     'attention_decoder/Attention_0/bias'),
    ('attention_decoder/AttnOutputProjection/weights',
     'attention_decoder/AttnOutputProjection/kernel'),
    ('attention_decoder/AttnOutputProjection/biases',
     'attention_decoder/AttnOutputProjection/bias'),
])

_RNN_SHARDED_NAME_REPLACEMENTS = collections.OrderedDict([
    ('LSTMCell/W_', 'lstm_cell/weights/part_'),
    ('BasicLSTMCell/Linear/Matrix_', 'basic_lstm_cell/weights/part_'),
    ('GRUCell/W_', 'gru_cell/weights/part_'),
    ('MultiRNNCell/Cell', 'multi_rnn_cell/cell_'),
])


def _rnn_name_replacement(var_name):
  for pattern in RNN_NAME_REPLACEMENTS:
    if pattern in var_name:
      old_var_name = var_name
      var_name = var_name.replace(pattern, RNN_NAME_REPLACEMENTS[pattern])
      logging.info('Converted: %s --> %s' % (old_var_name, var_name))
      break
  return var_name


def _rnn_name_replacement_sharded(var_name):
  for pattern in _RNN_SHARDED_NAME_REPLACEMENTS:
    if pattern in var_name:
      old_var_name = var_name
      var_name = var_name.replace(pattern,
                                  _RNN_SHARDED_NAME_REPLACEMENTS[pattern])
      logging.info('Converted: %s --> %s' % (old_var_name, var_name))
  return var_name


def _split_sharded_vars(name_shape_map):
  """Split shareded variables.

  Args:
    name_shape_map: A dict from variable name to variable shape.

  Returns:
    not_sharded: Names of the non-sharded variables.
    sharded: Names of the sharded variables.
  """
  sharded = []
  not_sharded = []
  for name in name_shape_map:
    if re.match(name, '_[0-9]+$'):
      if re.sub('_[0-9]+$', '_1', name) in name_shape_map:
        sharded.append(name)
      else:
        not_sharded.append(name)
    else:
      not_sharded.append(name)
  return not_sharded, sharded


def convert_names(checkpoint_from_path,
                  checkpoint_to_path,
                  write_v1_checkpoint=False):
  """Migrates the names of variables within a checkpoint.

  Args:
    checkpoint_from_path: Path to source checkpoint to be read in.
    checkpoint_to_path: Path to checkpoint to be written out.
    write_v1_checkpoint: Whether the output checkpoint will be in V1 format.

  Returns:
    A dictionary that maps the new variable names to the Variable objects.
    A dictionary that maps the old variable names to the new variable names.
  """
  with ops.Graph().as_default():
    logging.info('Reading checkpoint_from_path %s' % checkpoint_from_path)
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_from_path)
    name_shape_map = reader.get_variable_to_shape_map()
    not_sharded, sharded = _split_sharded_vars(name_shape_map)
    new_variable_map = {}
    conversion_map = {}
    for var_name in not_sharded:
      new_var_name = _rnn_name_replacement(var_name)
      tensor = reader.get_tensor(var_name)
      var = variables.Variable(tensor, name=var_name)
      new_variable_map[new_var_name] = var
      if new_var_name != var_name:
        conversion_map[var_name] = new_var_name
    for var_name in sharded:
      new_var_name = _rnn_name_replacement_sharded(var_name)
      var = variables.Variable(tensor, name=var_name)
      new_variable_map[new_var_name] = var
      if new_var_name != var_name:
        conversion_map[var_name] = new_var_name

    write_version = (saver_pb2.SaverDef.V1
                     if write_v1_checkpoint else saver_pb2.SaverDef.V2)
    saver = saver_lib.Saver(new_variable_map, write_version=write_version)

    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      logging.info('Writing checkpoint_to_path %s' % checkpoint_to_path)
      saver.save(sess, checkpoint_to_path)

  logging.info('Summary:')
  logging.info('  Converted %d variable name(s).' % len(new_variable_map))
  return new_variable_map, conversion_map


def main(_):
  convert_names(
      FLAGS.checkpoint_from_path,
      FLAGS.checkpoint_to_path,
      write_v1_checkpoint=FLAGS.write_v1_checkpoint)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument('checkpoint_from_path', type=str,
                      help='Path to source checkpoint to be read in.')
  parser.add_argument('checkpoint_to_path', type=str,
                      help='Path to checkpoint to be written out.')
  parser.add_argument('--write_v1_checkpoint', action='store_true',
                      help='Write v1 checkpoint')
  FLAGS, unparsed = parser.parse_known_args()

  app.run(main=main, argv=[sys.argv[0]] + unparsed)
