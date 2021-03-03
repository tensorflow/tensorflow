# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================
"""Tests for xla_sharding.Sharding class and associated module functions."""

from absl.testing import absltest
import numpy as np

from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops


class ShardingTest(absltest.TestCase):

  def test_sharding_is_default_constructable(self):
    sharding = xla_sharding.Sharding()
    self.assertIsNotNone(sharding)

  def test_sharding_factory_functions_can_return_sharding_objects(self):
    """Tests the various recommended ways to construct a Sharding object.

    This is the most minimal of tests, doesn't assert anything about the
    Sharding object produced by a given factory methods other than that it
    has the correct type.
    """
    self.assertIsInstance(xla_sharding.Sharding.replicate(),
                          xla_sharding.Sharding)
    self.assertIsInstance(xla_sharding.Sharding.manual(), xla_sharding.Sharding)
    self.assertIsInstance(
        xla_sharding.Sharding.assign_device(0), xla_sharding.Sharding)
    self.assertIsInstance(
        xla_sharding.Sharding.tile(np.ones([3], dtype=int)),
        xla_sharding.Sharding)
    self.assertIsInstance(
        xla_sharding.Sharding.partial_tile(np.ones([3], dtype=int)),
        xla_sharding.Sharding)
    self.assertIsInstance(
        xla_sharding.Sharding.split(
            array_ops.ones([3, 8, 7], dtype=dtypes.int32), 1, 2),
        xla_sharding.Sharding)


if __name__ == '__main__':
  absltest.main()
