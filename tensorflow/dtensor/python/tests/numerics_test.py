# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for numerics in DTensor Ops."""

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test

Layout = layout_lib.Layout
Mesh = layout_lib.Mesh
UNSHARDED = layout_lib.UNSHARDED
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_DIMS = [_MESH_DIM_X, _MESH_DIM_Y]


class NumericTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(NumericTest, self).setUp()

    self.skipForDeviceType(['TPU'],
                           'all tests require 8 TPU cores.',
                           unless_device_count_equals_to=8)

    test_util.reset_logical_devices('CPU', 8)
    accelerator_util.initialize_accelerator_system()

    self.stateless_random_seed = [0, 1]

  def _create_mesh(self, topology, device):
    device_ids = test_util.create_device_ids_array(topology)
    return Mesh(
        _MESH_DIMS,
        device_ids,
        np.ravel(device_ids).tolist(),
        test_util.create_device_list(topology, device),
    )

  # Tests AllReduce numerics with and without mixed precision reduce enabled,
  # based on go/dtensor-numerics.
  @parameterized.named_parameters(('_without_mixed_precision_reduce', False),
                                  ('_with_mixed_precision_reduce', True))
  def test_all_reduce(self, enable_mixed_precision_reduce):
    if enable_mixed_precision_reduce:
      os.environ['DTENSOR_ENABLE_MIXED_PRECISION_REDUCE'] = ''
      # Override group size since we are testing on smaller mesh.
      os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE'] = '4'
    else:
      if 'DTENSOR_ENABLE_MIXED_PRECISION_REDUCE' in os.environ:
        del os.environ['DTENSOR_ENABLE_MIXED_PRECISION_REDUCE']

    @polymorphic_function.function
    def _compute_reduction(inp):
      return math_ops.reduce_sum(inp, axis=[2])

    input_tensor = stateless_random_ops.stateless_random_uniform(
        shape=(8, 8, 8, 64),
        seed=self.stateless_random_seed,
        minval=-5.0,
        maxval=5.0,
        dtype=dtypes.bfloat16,
    )
    expected = _compute_reduction(input_tensor)

    # Compute reduction on 8x1, since dim 2 is unsharded AllReduce will not be
    # needed.
    mesh_8x1 = self._create_mesh((8, 1), 'TPU')
    input_8x1 = numpy_util.pack_numpy(
        input_tensor,
        Layout([_MESH_DIM_X, UNSHARDED, UNSHARDED, UNSHARDED], mesh_8x1),
    )
    result_8x1 = _compute_reduction(input_8x1)
    result_8x1_np = numpy_util.to_numpy(result_8x1)

    # Compute reduction on 1x8, AllReduce will be needed since dim 2 is sharded.
    mesh_1x8 = self._create_mesh((1, 8), 'TPU')
    input_1x8 = numpy_util.pack_numpy(
        input_tensor,
        Layout([_MESH_DIM_X, UNSHARDED, _MESH_DIM_Y, UNSHARDED], mesh_1x8),
    )
    result_1x8 = _compute_reduction(input_1x8)
    result_1x8_np = numpy_util.to_numpy(result_1x8)

    self.assertEqual(result_8x1.dtype, dtypes.bfloat16)
    self.assertEqual(result_1x8.dtype, dtypes.bfloat16)

    # Mixed precision does not apply since AllReduce was not used, result will
    # always be close to the expected value.
    self.assertAllClose(result_8x1_np, expected, atol=1e-5, rtol=1e-5)

    # AllReduce was needed, so result will be more accurate if mixed precision
    # is enabled.
    if enable_mixed_precision_reduce:
      self.assertAllClose(result_1x8_np, expected, atol=1e-5, rtol=1e-5)
    else:
      self.assertNotAllClose(result_1x8_np, expected, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  test.main()
