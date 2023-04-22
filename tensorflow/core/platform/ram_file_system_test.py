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

"""Tests for ram_file_system.h."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.layers import core as core_layers
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training import adam
from tensorflow.python.training import training_util


class RamFilesystemTest(test_util.TensorFlowTestCase):

  def test_create_and_delete_directory(self):
    file_io.create_dir_v2('ram://testdirectory')
    file_io.delete_recursively_v2('ram://testdirectory')

  def test_create_and_delete_directory_tree_recursive(self):
    file_io.create_dir_v2('ram://testdirectory')
    file_io.create_dir_v2('ram://testdirectory/subdir1')
    file_io.create_dir_v2('ram://testdirectory/subdir2')
    file_io.create_dir_v2('ram://testdirectory/subdir1/subdir3')
    with gfile.GFile('ram://testdirectory/subdir1/subdir3/a.txt', 'w') as f:
      f.write('Hello, world.')
    file_io.delete_recursively_v2('ram://testdirectory')
    self.assertEqual(gfile.Glob('ram://testdirectory/*'), [])

  def test_write_file(self):
    with gfile.GFile('ram://a.txt', 'w') as f:
      f.write('Hello, world.')
      f.write('Hello, world.')

    with gfile.GFile('ram://a.txt', 'r') as f:
      self.assertEqual(f.read(), 'Hello, world.' * 2)

  def test_append_file_with_seek(self):
    with gfile.GFile('ram://c.txt', 'w') as f:
      f.write('Hello, world.')

    with gfile.GFile('ram://c.txt', 'w+') as f:
      f.seek(offset=0, whence=2)
      f.write('Hello, world.')

    with gfile.GFile('ram://c.txt', 'r') as f:
      self.assertEqual(f.read(), 'Hello, world.' * 2)

  def test_list_dir(self):
    for i in range(10):
      with gfile.GFile('ram://a/b/%d.txt' % i, 'w') as f:
        f.write('')
      with gfile.GFile('ram://c/b/%d.txt' % i, 'w') as f:
        f.write('')

    matches = ['%d.txt' % i for i in range(10)]
    self.assertEqual(gfile.ListDirectory('ram://a/b/'), matches)

  def test_glob(self):
    for i in range(10):
      with gfile.GFile('ram://a/b/%d.txt' % i, 'w') as f:
        f.write('')
      with gfile.GFile('ram://c/b/%d.txt' % i, 'w') as f:
        f.write('')

    matches = ['ram://a/b/%d.txt' % i for i in range(10)]
    self.assertEqual(gfile.Glob('ram://a/b/*'), matches)

    matches = []
    self.assertEqual(gfile.Glob('ram://b/b/*'), matches)

    matches = ['ram://c/b/%d.txt' % i for i in range(10)]
    self.assertEqual(gfile.Glob('ram://c/b/*'), matches)

  def test_file_exists(self):
    with gfile.GFile('ram://exists/a/b/c.txt', 'w') as f:
      f.write('')
    self.assertTrue(gfile.Exists('ram://exists/a'))
    self.assertTrue(gfile.Exists('ram://exists/a/b'))
    self.assertTrue(gfile.Exists('ram://exists/a/b/c.txt'))

    self.assertFalse(gfile.Exists('ram://exists/b'))
    self.assertFalse(gfile.Exists('ram://exists/a/c'))
    self.assertFalse(gfile.Exists('ram://exists/a/b/k'))

  def test_estimator(self):

    def model_fn(features, labels, mode, params):
      del params
      x = core_layers.dense(features, 100)
      x = core_layers.dense(x, 100)
      x = core_layers.dense(x, 100)
      x = core_layers.dense(x, 100)
      y = core_layers.dense(x, 1)
      loss = losses.mean_squared_error(labels, y)
      opt = adam.AdamOptimizer(learning_rate=0.1)
      train_op = opt.minimize(
          loss, global_step=training_util.get_or_create_global_step())

      return EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def input_fn():
      batch_size = 128
      return (constant_op.constant(np.random.randn(batch_size, 100),
                                   dtype=dtypes.float32),
              constant_op.constant(np.random.randn(batch_size, 1),
                                   dtype=dtypes.float32))

    config = RunConfig(
        model_dir='ram://estimator-0/', save_checkpoints_steps=1)
    estimator = Estimator(config=config, model_fn=model_fn)

    estimator.train(input_fn=input_fn, steps=10)
    estimator.train(input_fn=input_fn, steps=10)
    estimator.train(input_fn=input_fn, steps=10)
    estimator.train(input_fn=input_fn, steps=10)

  def test_savedmodel(self):
    class MyModule(module.Module):

      @def_function.function(input_signature=[])
      def foo(self):
        return constant_op.constant([1])

    saved_model.save(MyModule(), 'ram://my_module')

    loaded = saved_model.load('ram://my_module')
    self.assertAllEqual(loaded.foo(), [1])


if __name__ == '__main__':
  test.main()
