# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPUReplicatedVariable."""
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.eager import test
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as variables_lib


class TPUReplicatedVariableTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_tpu_replicated_variable_simple(self):
    v0 = variables_lib.Variable([0], name='v0')
    v1 = variables_lib.Variable([0], name='v1')
    r = tpu_replicated_variable.TPUReplicatedVariable([v0, v1])
    self.evaluate(variables_lib.global_variables_initializer())
    self.assertEqual(r.variables[0], v0)
    self.assertEqual(r.variables[1], v1)
    self.assertEqual(r.shape.as_list(), [1])
    self.assertEqual(r.dtype, v0.dtype)
    self.check_replicated_variables_all_the_same(r)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_tpu_replicated_variable_update(self):
    batch_size = 32
    num_feature_in = 16

    x = np.random.rand(batch_size, num_feature_in).astype(np.float32)
    w_init = np.random.rand(batch_size, num_feature_in).astype(np.float32)

    w0 = variables_lib.Variable(w_init, dtype=dtypes.float32, name='w0')
    w1 = variables_lib.Variable(w_init, dtype=dtypes.float32, name='w1')
    self.evaluate(variables_lib.global_variables_initializer())
    w = tpu_replicated_variable.TPUReplicatedVariable([w0, w1])

    # Make a copy of x so that `w` and `x` do not share the same buffer.
    # See b/195972684.
    self.evaluate(w.assign(x.copy()))
    result = self.evaluate(w.read_value())
    self.assertAllClose(result, x)
    self.check_replicated_variables_all_the_same(w)

    x1 = np.random.rand(batch_size, num_feature_in).astype(np.float32)
    self.evaluate(w.assign_sub(x1))
    result = self.evaluate(w.read_value())
    self.assertAllClose(result, np.subtract(x, x1))
    self.check_replicated_variables_all_the_same(w)

    x2 = np.random.rand(batch_size, num_feature_in).astype(np.float32)
    self.evaluate(w.assign(x.copy()))
    self.evaluate(w.assign_add(x2))
    result = self.evaluate(w.read_value())
    self.assertAllClose(result, np.add(x, x2))
    self.check_replicated_variables_all_the_same(w)

  def check_replicated_variables_all_the_same(self, rv):
    for v in rv.variables:
      self.assertAllEqual(
          self.evaluate(rv.variables[0].read_value()),
          self.evaluate(v))

  @combinations.generate(combinations.combine(
      mode=['graph', 'eager'],
      enable_async_ckpt=[True, False]
  ))
  def test_tpu_replicated_variable_checkpoint(self, enable_async_ckpt):
    batch_size = 4
    num_feature_in = 2

    # Initialize variables
    x = np.random.rand(batch_size, num_feature_in).astype(np.float32)
    w_init = np.random.rand(batch_size, num_feature_in).astype(np.float32)

    w0 = variables_lib.Variable(w_init, dtype=dtypes.float32, name='w0')
    w1 = variables_lib.Variable(w_init, dtype=dtypes.float32, name='w1')
    self.evaluate(variables_lib.global_variables_initializer())
    w = tpu_replicated_variable.TPUReplicatedVariable([w0, w1])
    before_save = self.evaluate(w.read_value())

    # Save w_init into checkpoint
    ckpt = trackable_utils.Checkpoint(w=w)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    with self.test_session():
      save_path = ckpt.save(file_prefix=prefix, options=ckpt_options)

    # Change values of w to x
    self.evaluate(w.assign(x.copy()))
    result = self.evaluate(w.read_value())
    self.assertAllClose(result, x)
    self.check_replicated_variables_all_the_same(w)

    # Restore from the checkpoint
    with self.test_session():
      ckpt.restore(save_path).assert_consumed().run_restore_ops()
    after_restore = self.evaluate(w.read_value())
    self.check_replicated_variables_all_the_same(w)
    self.assertAllClose(before_save, after_restore)

    # Another round of saving/restoring to ensure that the logic of
    # _copy_trackable_to_cpu works when a copy is already created in object_map.
    y = np.random.rand(batch_size, num_feature_in).astype(np.float32)
    z = np.random.rand(batch_size, num_feature_in).astype(np.float32)
    self.evaluate(w.assign(y.copy()))  # change from x to y
    before_save = self.evaluate(w.read_value())
    self.assertAllClose(before_save, y)
    self.check_replicated_variables_all_the_same(w)

    with self.test_session():
      save_path = ckpt.save(file_prefix=prefix, options=ckpt_options)

    self.evaluate(w.assign(z.copy()))  # change from y to z
    result = self.evaluate(w.read_value())
    self.assertAllClose(result, z)

    with self.test_session():
      ckpt.restore(save_path).assert_consumed().run_restore_ops()
    after_restore = self.evaluate(w.read_value())
    self.check_replicated_variables_all_the_same(w)
    self.assertAllClose(before_save, after_restore)


if __name__ == '__main__':
  test.main()
