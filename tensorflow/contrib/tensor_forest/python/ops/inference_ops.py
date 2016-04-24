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
"""Ops for BrainTree v2 tree evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

INFERENCE_OPS_FILE = '_inference_ops.so'

_inference_ops = None
_ops_lock = threading.Lock()


ops.NoGradient('TreePredictions')


@ops.RegisterShape('TreePredictions')
def TreePredictions(op):
  """Shape function for TreePredictions Op."""
  num_points = op.inputs[0].get_shape()[0].value
  num_classes = op.inputs[3].get_shape()[1].value
  # The output of TreePredictions is
  # [node_pcw(evaluate_tree(x), c) for c in classes for x in input_data].
  return [tensor_shape.TensorShape([num_points, num_classes])]


# Workaround for the fact that importing tensorflow imports contrib
# (even if a user isn't using this or any other contrib op), but
# there's not yet any guarantee that the shared object exists.
# In which case, "import tensorflow" will always crash, even for users that
# never use contrib.
def Load(library_base_dir=''):
  """Load the inference ops library and return the loaded module."""
  with _ops_lock:
    global _inference_ops
    if not _inference_ops:
      data_files_path = os.path.join(library_base_dir,
                                     tf.resource_loader.get_data_files_path())
      tf.logging.info('data path: %s', data_files_path)
      _inference_ops = tf.load_op_library(os.path.join(
          data_files_path, INFERENCE_OPS_FILE))

      assert _inference_ops, 'Could not load inference_ops.so'
  return _inference_ops
