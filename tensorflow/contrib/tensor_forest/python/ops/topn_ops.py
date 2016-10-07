# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for TopN class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import tensorflow as tf

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops

TOPN_OPS_FILE = '_topn_ops.so'

_topn_ops = None
_ops_lock = threading.Lock()

ops.NotDifferentiable('TopNInsert')
ops.NotDifferentiable('TopNRemove')


ops.RegisterShape('TopNInsert')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('TopNRemove')(common_shapes.call_cpp_shape_fn)


# Workaround for the fact that importing tensorflow imports contrib
# (even if a user isn't using this or any other contrib op), but
# there's not yet any guarantee that the shared object exists.
# In which case, "import tensorflow" will always crash, even for users that
# never use contrib.
def Load():
  """Load the TopN ops library and return the loaded module."""
  with _ops_lock:
    global _topn_ops
    if not _topn_ops:
      ops_path = tf.resource_loader.get_path_to_datafile(TOPN_OPS_FILE)
      tf.logging.info('data path: %s', ops_path)
      _topn_ops = tf.load_op_library(ops_path)

      assert _topn_ops, 'Could not load topn_ops.so'
  return _topn_ops
