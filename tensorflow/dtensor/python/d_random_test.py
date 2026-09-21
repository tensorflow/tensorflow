# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for seed dtype enforcement in d_random stateless RNG functions."""

from tensorflow.dtensor.python import d_random
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops


class SeedDtypeEnforcementTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()
    self.mesh = self.configTestMesh({"CPU": test_util.create_device_list((2,), "CPU")})
    self.layout = layout_lib.Layout.replicated(self.mesh, rank=2)

  def test_stateless_random_uniform_rejects_incompatible_seed_dtype(self):
    seed = ops.convert_to_tensor([1, 2], dtype=dtypes.int64)
    with self.assertRaises((ValueError, TypeError)):
      d_random._old_tf_random_stateless_uniform(
          shape=[2, 2], seed=seed, layout=self.layout, minval=0, maxval=1,
          dtype=dtypes.float32)

  def test_stateless_random_normal_rejects_incompatible_seed_dtype(self):
    seed = ops.convert_to_tensor([1, 2], dtype=dtypes.int64)
    with self.assertRaises((ValueError, TypeError)):
      d_random._old_tf_random_stateless_normal(
          shape=[2, 2], seed=seed, layout=self.layout, dtype=dtypes.float32)

  def test_stateless_truncated_normal_rejects_incompatible_seed_dtype(self):
    seed = ops.convert_to_tensor([1, 2], dtype=dtypes.int64)
    with self.assertRaises((ValueError, TypeError)):
      d_random._old_tf_stateless_truncated_normal(
          shape=[2, 2], seed=seed, layout=self.layout, dtype=dtypes.float32)

  def test_stateless_random_uniform_accepts_int32_seed(self):
    seed = ops.convert_to_tensor([1, 2], dtype=dtypes.int32)
    result = d_random._old_tf_random_stateless_uniform(
        shape=[2, 2], seed=seed, layout=self.layout, minval=0, maxval=1,
        dtype=dtypes.float32)
    self.assertEqual(result.shape, [2, 2])

  def test_stateless_random_uniform_accepts_plain_list_seed(self):
    result = d_random._old_tf_random_stateless_uniform(
        shape=[2, 2], seed=[1, 2], layout=self.layout, minval=0, maxval=1,
        dtype=dtypes.float32)
    self.assertEqual(result.shape, [2, 2])


if __name__ == "__main__":
  test_util.main()
