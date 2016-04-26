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
"""Ops for quantized evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading

import tensorflow as tf

QUANTIZED_OPS_FILE = '_quantized_ops.so'

_quantized_ops = None
_ops_lock = threading.Lock()


# Workaround for the fact that importing tensorflow imports contrib
# (even if a user isn't using this or any other contrib op), but
# there's not yet any guarantee that the shared object exists.
# In which case, "import tensorflow" will always crash, even for users that
# never use contrib.
def Load(library_base_dir=''):
  """Load the quantized ops library and return the loaded module."""
  with _ops_lock:
    global _quantized_ops
    if not _quantized_ops:
      data_files_path = os.path.join(library_base_dir,
                                     tf.resource_loader.get_data_files_path())
      tf.logging.info('q:data path: %s', data_files_path)
      _quantized_ops = tf.load_op_library(os.path.join(
          data_files_path, QUANTIZED_OPS_FILE))

      assert _quantized_ops, 'Could not load quantized_ops.so'
  return _quantized_ops
