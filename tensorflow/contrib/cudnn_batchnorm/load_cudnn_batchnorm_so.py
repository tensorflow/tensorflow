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
"""Ops for cudnn batchnorm."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading

import tensorflow as tf

CUDNN_BATCHNORM_FILE = '_cudnn_batchnorm.so'

_cudnn_batchnorm = None
_lock = threading.Lock()

def Load(library_base_dir=''):
  """Load the cudnn batchnorm ops library and return the loaded module."""
  with _lock:
    global _cudnn_batchnorm
    if not _cudnn_batchnorm:
      data_files_path = os.path.join(library_base_dir,
                                     tf.resource_loader.get_data_files_path())
      tf.logging.info('data path: %s', data_files_path)
      _cudnn_batchnorm = tf.load_op_library(os.path.join(
          data_files_path, CUDNN_BATCHNORM_FILE))

      assert _cudnn_batchnorm, 'Could not load _cudnn_batchnorm.so'
  return _cudnn_batchnorm
