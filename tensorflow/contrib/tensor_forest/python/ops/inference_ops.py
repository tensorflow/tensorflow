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
"""Ops for BrainTree v2 tree evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging


INFERENCE_OPS_FILE = '_inference_ops.so'

_inference_ops = None
_ops_lock = threading.Lock()


# TODO(b/31222613): This op may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable('TreePredictions')


# Workaround for the fact that importing tensorflow imports contrib
# (even if a user isn't using this or any other contrib op), but
# there's not yet any guarantee that the shared object exists.
# In which case, "import tensorflow" will always crash, even for users that
# never use contrib.
def Load():
  """Load the inference ops library and return the loaded module."""
  with _ops_lock:
    global _inference_ops
    if not _inference_ops:
      ops_path = resource_loader.get_path_to_datafile(INFERENCE_OPS_FILE)
      logging.info('data path: %s', ops_path)
      _inference_ops = load_library.load_op_library(ops_path)

      assert _inference_ops, 'Could not load inference_ops.so'
  return _inference_ops
