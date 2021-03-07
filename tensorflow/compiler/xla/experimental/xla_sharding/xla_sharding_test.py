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

from google.protobuf.message import DecodeError
from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops


class ShardingTest(test_util.TensorFlowTestCase):
  """Tests for member functions of the class xla_sharding.Sharding."""

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


class XlaShardingTest(test_util.TensorFlowTestCase):
  """Tests for non-member functions in the module xla_sharding.py."""

  def test_replicate_annotates_tensor_correctly(self):

    @def_function.function
    def replicate_helper(tensor):
      replicated_tensor = xla_sharding.replicate(
          array_ops.ones([4, 5, 6], dtype=dtypes.float32))
      self.assertIsNone(xla_sharding.get_tensor_sharding(tensor))
      replicated_sharding = xla_sharding.get_tensor_sharding(replicated_tensor)
      self.assertIsNotNone(replicated_sharding)
      self.assertIsNone(
          xla_sharding.get_sharding_tile_shape(replicated_sharding))
      return replicated_tensor

    in_tensor = array_ops.ones([4, 5, 6], dtype=dtypes.float32)
    result = replicate_helper(array_ops.ones([4, 5, 6], dtype=dtypes.float32))
    self.assertAllEqual(in_tensor, result)

  def test_tile_annotates_tensor_correctly(self):

    @def_function.function
    def tile_helper(tensor):
      self.assertIsNone(xla_sharding.get_tensor_sharding(tensor))
      tiled_tensor = xla_sharding.tile(tensor, np.array([2, 1, 6]))
      self.assertIsInstance(tiled_tensor, ops.Tensor)
      tiled_sharding = xla_sharding.get_tensor_sharding(tiled_tensor)
      tile_shape = xla_sharding.get_sharding_tile_shape(tiled_sharding)
      # This is the shape of the tile assignment [2, 1, 6]
      expected_shape = [3]
      self.assertEqual(expected_shape, tile_shape)
      return tiled_tensor

    in_tensor = array_ops.ones([4, 5, 6], dtype=dtypes.float32)
    result = tile_helper(array_ops.ones([4, 5, 6], dtype=dtypes.float32))
    self.assertAllEqual(in_tensor, result)

  def test_split_annotates_tensor_correctly(self):

    @def_function.function
    def split_helper(tensor):
      self.assertIsNone(xla_sharding.get_tensor_sharding(tensor))
      split_tensor = xla_sharding.split(tensor, 2, 3)
      self.assertIsInstance(split_tensor, ops.Tensor)
      split_sharding = xla_sharding.get_tensor_sharding(split_tensor)
      split_shape = xla_sharding.get_sharding_tile_shape(split_sharding)
      expected_shape = [1, 1, 3]
      self.assertEqual(expected_shape, split_shape)
      return split_tensor

    in_tensor = array_ops.ones([4, 5, 6], dtype=dtypes.float32)
    result = split_helper(
        array_ops.ones([4, 5, 6], dtype=dtypes.float32))
    self.assertAllEqual(in_tensor, result)

  def test_split_raises_error_with_incommensurate_dimensions(self):

    @def_function.function
    def split_helper(tensor):
      split_tensor = xla_sharding.split(tensor, 0, 8)
      return split_tensor

    with self.assertRaises(ValueError):
      _ = split_helper(array_ops.ones([4, 5, 6], dtype=dtypes.float32))

    # TODO(drm): Modify split() so that this call raises an error since
    # 8 does not divide 9 (currently only checks that 8 is smaller than 9,
    # which it is, but this is not good for splitting).
    # with self.assertRaises(ValueError):
    #   _ = split_helper(array_ops.ones([9, 5, 6], dtype=dtypes.float32))

  def test_copy_sharding_succeeds_with_identically_shaped_tensors(self):

    @def_function.function
    def copy_helper(tensor):
      tensor_src = array_ops.identity(tensor)
      tensor_src = xla_sharding.split(tensor, 2, 3)
      sharding_src = xla_sharding.get_tensor_sharding(tensor_src)
      shape_src = xla_sharding.get_sharding_tile_shape(sharding_src)
      self.assertEqual([1, 1, 3], shape_src)

      tensor_dest = array_ops.identity(tensor)
      self.assertIsNone(xla_sharding.get_tensor_sharding(tensor_dest))

      xla_sharding.copy_sharding(tensor_src, tensor_dest)
      sharding_dest = xla_sharding.get_tensor_sharding(tensor_dest)
      shape_dest = xla_sharding.get_sharding_tile_shape(sharding_dest)
      self.assertEqual([1, 1, 3], shape_dest)
      return tensor_dest

    in_tensor = array_ops.ones([4, 5, 6], dtype=dtypes.float32)
    result = copy_helper(
        array_ops.ones([4, 5, 6], dtype=dtypes.float32))
    self.assertAllEqual(in_tensor, result)

  def test_get_sharding_tile_shape_returns_none_on_none_input(self):
    self.assertIsNone(xla_sharding.get_sharding_tile_shape(None))

  def test_get_sharding_tile_shape_raises_error_on_nonparsable_input(self):
    bad_proto_data = b'\x0f'
    with self.assertRaises(DecodeError):
      xla_sharding.get_sharding_tile_shape(bad_proto_data)


if __name__ == '__main__':
  absltest.main()
