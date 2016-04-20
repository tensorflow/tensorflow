# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Ops for BrainTree v2 training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


TRAINING_OPS_FILE = '_training_ops.so'

_training_ops = None
_ops_lock = threading.Lock()

ops.NoGradient('CountExtremelyRandomStats')
ops.NoGradient('SampleInputs')
ops.NoGradient('BestSplits')
ops.NoGradient('GrowTree')
ops.NoGradient('FinishedNodes')
ops.NoGradient('ScatterAddNdim')
ops.NoGradient('UpdateFertileSlots')


@ops.RegisterShape('CountExtremelyRandomStats')
def _CountExtremelyRandomStatsShape(op):
  """Shape function for CountExtremelyRandomStats Op."""
  num_points = op.inputs[0].get_shape()[0].value
  num_nodes = op.inputs[2].get_shape()[0].value
  num_classes = op.get_attr('num_classes')
  # The output of TraverseTree is [leaf_node_index(x) for x in input_data].
  return [tensor_shape.TensorShape([num_nodes, num_classes]),  # node pcw
          tensor_shape.TensorShape([None, 3]),
          tensor_shape.TensorShape([None]),
          tensor_shape.TensorShape([None, 2]),
          tensor_shape.TensorShape([None]),
          tensor_shape.TensorShape([num_points])]


@ops.RegisterShape('SampleInputs')
def _SampleInputsShape(op):
  """Shape function for SampleInputs Op."""
  num_splits = op.inputs[3].get_shape()[1].value
  return [[None], [None, num_splits], [None, num_splits]]


@ops.RegisterShape('BestSplits')
def _BestSplitsShape(op):
  num_finished = op.inputs[0].get_shape()[0].value
  return [tensor_shape.TensorShape([num_finished])]


@ops.RegisterShape('GrowTree')
def _GrowTreeShape(unused_op):
  """Shape function for GrowTree Op."""
  return [[None], [None, 2], [None], [None], [1]]


@ops.RegisterShape('FinishedNodes')
def _FinishedNodesShape(unused_op):
  """Shape function for FinishedNodes Op."""
  return [[None]]


@ops.RegisterShape('ScatterAddNdim')
def _ScatterAddNdimShape(unused_op):
  """Shape function for ScatterAddNdim Op."""
  return []


@ops.RegisterShape('UpdateFertileSlots')
def _UpdateFertileSlotsShape(unused_op):
  """Shape function for UpdateFertileSlots Op."""
  return [[None, 2], [None], [None], [None], [None]]


# Workaround for the fact that importing tensorflow imports contrib
# (even if a user isn't using this or any other contrib op), but
# there's not yet any guarantee that the shared object exists.
# In which case, "import tensorflow" will always crash, even for users that
# never use contrib.
def Load(library_base_dir=''):
  """Load training ops library and return the loaded module."""
  with _ops_lock:
    global _training_ops
    if not _training_ops:
      data_files_path = os.path.join(library_base_dir,
                                     tf.resource_loader.get_data_files_path())
      tf.logging.info('data path: %s', data_files_path)
      _training_ops = tf.load_op_library(os.path.join(
          data_files_path, TRAINING_OPS_FILE))

      assert _training_ops, 'Could not load _training_ops.so'
  return _training_ops
