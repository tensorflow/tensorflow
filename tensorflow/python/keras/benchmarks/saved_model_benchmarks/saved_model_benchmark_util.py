# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utils for saved model benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import time

import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


def save_and_load_benchmark(app):
  """Util for saved model benchmarks."""
  trials = 3

  model = app(weights=None)
  model_name = app.__name__

  tmp_dir = test.get_temp_dir()
  gfile.MakeDirs(tmp_dir)
  save_dir = tempfile.mkdtemp(dir=tmp_dir)

  total_save_time = 0
  total_load_time = 0

  # Run one untimed iteration of saving/loading.
  model.save(save_dir, save_format='tf')
  tf.keras.models.load_model(save_dir)

  for _ in range(trials):
    start_time = time.time()
    model.save(save_dir, save_format='tf')
    total_save_time += time.time() - start_time

    start_time = time.time()
    tf.keras.models.load_model(save_dir)
    total_load_time += time.time() - start_time

  save_result = {
      'iters': trials,
      'wall_time': total_save_time / trials,
      'name': '{}.save'.format(model_name)
  }

  load_result = {
      'iters': trials,
      'wall_time': total_load_time / trials,
      'name': '{}.load'.format(model_name)
  }
  gfile.DeleteRecursively(save_dir)
  return save_result, load_result

