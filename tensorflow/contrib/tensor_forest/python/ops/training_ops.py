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
"""Ops for BrainTree v2 training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging


TRAINING_OPS_FILE = '_training_ops.so'

_training_ops = None
_ops_lock = threading.Lock()

ops.NotDifferentiable('CountExtremelyRandomStats')
ops.NotDifferentiable('SampleInputs')
ops.NotDifferentiable('BestSplits')
ops.NotDifferentiable('GrowTree')
ops.NotDifferentiable('FinishedNodes')
# TODO(b/31222613): This op may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable('ScatterAddNdim')
ops.NotDifferentiable('UpdateFertileSlots')


ops.RegisterShape('CountExtremelyRandomStats')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('SampleInputs')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('BestSplits')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('GrowTree')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('FinishedNodes')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('ScatterAddNdim')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('UpdateFertileSlots')(common_shapes.call_cpp_shape_fn)


# Workaround for the fact that importing tensorflow imports contrib
# (even if a user isn't using this or any other contrib op), but
# there's not yet any guarantee that the shared object exists.
# In which case, "import tensorflow" will always crash, even for users that
# never use contrib.
def Load():
  """Load training ops library and return the loaded module."""
  with _ops_lock:
    global _training_ops
    if not _training_ops:
      ops_path = resource_loader.get_path_to_datafile(TRAINING_OPS_FILE)
      logging.info('data path: %s', ops_path)
      _training_ops = load_library.load_op_library(ops_path)

      assert _training_ops, 'Could not load _training_ops.so'
  return _training_ops
