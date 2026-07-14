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
"""Tests DTensor device cache for compiled function computation."""

import gc
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

# Convenient constants to use for tests.
_BATCH_DIM = "batch"
_MESH_DIM_X = "x"

# Shorter notation.
Layout = layout_lib.Layout
Mesh = layout_lib.Mesh


def diff_dicts(dict1, dict2):
  keys = set(dict1.keys()) | set(dict2.keys())
  return {key: dict1.get(key, 0) - dict2.get(key, 0) for key in keys}


class DTensorDeviceCacheTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorDeviceCacheTest, self).setUp()
    device_ids = test_util.create_device_ids_array((2,))
    local_device_ids = np.ravel(device_ids).tolist()
    mesh_dict = {
        device: Mesh(
            [_BATCH_DIM],
            device_ids,
            local_device_ids,
            test_util.create_device_list((2,), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def testBasic(self):

    @polymorphic_function.function
    def func0(a):
      return a + 1

    @polymorphic_function.function
    def func1(a):
      return a + 2

    c0 = api.copy_to_mesh(
        constant_op.constant(1.0), Layout.replicated(self.mesh, rank=0)
    )
    c1 = api.copy_to_mesh(
        constant_op.constant([2.0, 3.0]), Layout.replicated(self.mesh, rank=1)
    )
    c2 = api.copy_to_mesh(
        constant_op.constant([4.0]), Layout.replicated(self.mesh, rank=1)
    )
    c3 = api.copy_to_mesh(
        constant_op.constant(1, dtype=dtypes.int32),
        Layout.replicated(self.mesh, rank=0),
    )

    # c0 and c1 have different layouts. c1 and c2 have different shapes.
    # c0 and c3 have different dtypes.
    self.assertAllEqual(func0(c0), 2.0)
    self.assertAllEqual(func0(c1), [3.0, 4.0])
    self.assertAllEqual(func0(c2), [5.0])
    self.assertAllEqual(func0(c3), 2)

    # func0 and func1 have different names.
    self.assertAllEqual(func1(c0), 3.0)

  def testFunctionInputConstantFoldingCacheHits(self):

    @polymorphic_function.function
    def add(a, b):
      return a + b

    c0 = api.copy_to_mesh(
        constant_op.constant(17.0), Layout.replicated(self.mesh, rank=0)
    )
    c1 = api.copy_to_mesh(
        constant_op.constant(21.0), Layout.replicated(self.mesh, rank=0)
    )

    stats1 = api._dtensor_device()._get_stats()
    self.assertAllEqual(add(c0, c1), 38.0)
    self.assertAllEqual(add(c0, c1), 38.0)

    # First call should miss and second should hit.
    stats2 = api._dtensor_device()._get_stats()
    diff = {key: stats2[key] - stats1[key] for key in stats1.keys()}
    self.assertEqual(diff["function_manager.miss"], 1)
    self.assertEqual(diff["function_manager.hit"], 1)

  def testFunctionInputConstantFoldingCacheMiss(self):

    @polymorphic_function.function
    def add(a, b):
      return a + b

    c0 = api.copy_to_mesh(
        constant_op.constant(17.0), Layout.replicated(self.mesh, rank=0)
    )
    c1 = api.copy_to_mesh(
        constant_op.constant(21.0), Layout.replicated(self.mesh, rank=0)
    )
    c2 = api.copy_to_mesh(
        constant_op.constant(0.0), Layout.replicated(self.mesh, rank=0)
    )

    stats1 = api._dtensor_device()._get_stats()
    # First call should log a cache miss.
    self.assertAllEqual(add(c0, c1), 38.0)

    # Second call should also log a cache miss since second constant changed.
    self.assertAllEqual(add(c0, c2), 17.0)

    # Third call should not log a cache miss since the same input as the prev.
    self.assertAllEqual(add(c0, c2), 17.0)

    # Fourth call should log a cache miss since first input changed.
    self.assertAllEqual(add(c1, c2), 21.0)

    stats2 = api._dtensor_device()._get_stats()
    diff = {key: stats2[key] - stats1[key] for key in stats1.keys()}
    self.assertEqual(diff["function_manager.miss"], 3)
    self.assertEqual(diff["function_manager.hit"], 1)

  def testCacheWithRNG(self):
    with api._dtensor_device()._default_layout(
        Layout.replicated(self.mesh, rank=1)):
      v0 = gen_stateless_random_ops.stateless_random_normal(
          shape=[1], seed=[1, 2]
      )

    with api._dtensor_device()._default_layout(
        Layout.replicated(self.mesh, rank=1)):
      v1 = gen_stateless_random_ops.stateless_random_normal(
          shape=[1], seed=[1, 2]
      )
      v2 = gen_stateless_random_ops.stateless_random_normal(
          shape=[2], seed=[1, 2]
      )
      v3 = gen_stateless_random_ops.stateless_random_normal(
          shape=[1], seed=[3, 4]
      )

    # v0 and v1 have same layouts.
    self.assertAllEqual(v0, v1)
    api.check_layout(v0, Layout.replicated(self.mesh, rank=1))
    api.check_layout(v1, Layout.replicated(self.mesh, rank=1))
    # v1 and v2 have different shapes.
    self.assertNotEqual(v1.shape, v2.shape)
    # v1 and v3 have different seeds.
    self.assertNotEqual(v1.numpy(), v3.numpy())

  def testCacheWithVariable(self):
    c0 = api.copy_to_mesh(
        constant_op.constant(1.0), Layout.replicated(self.mesh, rank=0)
    )
    c1 = api.copy_to_mesh(
        constant_op.constant([2.0, 3.0]), Layout.replicated(self.mesh, rank=1)
    )
    a = constant_op.constant([4.0])
    b = constant_op.constant([5.0])
    c2 = api.pack(
        [a, b], layout=Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1)
    )

    v0 = d_variable.DVariable(c0)
    v1 = d_variable.DVariable(c1)
    v2 = d_variable.DVariable(c2)

    self.assertAllEqual(v0.read_value(), 1.0)
    self.assertAllEqual(v1.read_value(), [2.0, 3.0])
    unpacked_tensor = api.unpack(v2.read_value())
    self.assertAllClose([4.0], unpacked_tensor[0])
    self.assertAllClose([5.0], unpacked_tensor[1])

  @combinations.generate(
      combinations.combine(size=[16, 40], same_value=[True, False])
  )
  def testManyFunctions(self, size, same_value):
    r = range(100)

    values = [np.reshape(r[i : i + size], (4, size // 4)) for i in range(10)]
    c_layout = Layout.replicated(self.mesh, rank=2)
    values = [constant_op.constant(v, dtype=dtypes.float32) for v in values]
    c0 = [api.copy_to_mesh(v, c_layout) for v in values]

    c0 = [c0[0 if same_value else i] for i in range(10)]
    e0 = [values[0 if same_value else i] for i in range(10)]
    stats1 = api._dtensor_device()._get_stats()

    for i in range(10):
      # Use a special to ensure no conflicts with otherwise used names.
      @polymorphic_function.function
      def fn_31415926(c):
        return math_ops.reduce_sum(c)

      self.assertAllEqual(fn_31415926(c0[i]).numpy(), np.sum(e0[i]))

    del fn_31415926
    gc.collect()

    stats2 = api._dtensor_device()._get_stats()
    diff = diff_dicts(stats2, stats1)
    self.assertEqual(diff["function_manager.size"], 0)
    self.assertEqual(diff["kernel_cache.size"], 0)
    self.assertEqual(diff["device_cache.size"], 0)

  @combinations.generate(
      combinations.combine(size=[16, 40], same_value=[True, False])
  )
  def testManyEagerOps(self, size, same_value):
    if self.mesh.device_type() != "TPU":
      # For the CPU/GPU mesh, we have a shortcut that doesn't go through the
      # MLIR, but run the eager op locally and broadcast to all the devices.
      expected_cache_diff = 0
      expected_kernel_cache = 0
      expected_device_cache = 0
      expected_eager_pure_hit = 10
    else:
      # TODO(b/287529295): Remove this branch after the TPU issue is fixed.
      expected_device_cache = 0
      expected_eager_pure_hit = 0
      if same_value:
        expected_cache_diff = 1
        expected_kernel_cache = 2
      else:
        if size >= 20:
          expected_cache_diff = 1
          expected_kernel_cache = 2
        else:
          expected_cache_diff = 2
          expected_kernel_cache = 4

    r = range(100)
    c_layout = Layout.replicated(self.mesh, rank=2)
    values = [np.reshape(r[i : i + size], (4, size // 4)) for i in range(10)]
    values = [constant_op.constant(v, dtype=dtypes.float32) for v in values]
    c0 = [api.copy_to_mesh(v, c_layout) for v in values]

    c0 = [c0[0 if same_value else i] for i in range(10)]
    e0 = [values[0 if same_value else i] for i in range(10)]

    stats1 = api._dtensor_device()._get_stats()

    for i in range(10):
      self.assertAllEqual(array_ops.identity(c0[i]).numpy(), e0[i])

    gc.collect()

    stats2 = api._dtensor_device()._get_stats()
    diff = diff_dicts(stats2, stats1)

    if same_value:
      self.assertEqual(diff["function_manager.size"], expected_cache_diff)
      self.assertEqual(
          diff["eager_pure_optimization.hit"], expected_eager_pure_hit
      )
      # TFRT doesn't use eager cache.
      if not test_util.is_tfrt_enabled():
        self.assertEqual(diff["kernel_cache.size"], expected_kernel_cache)
        self.assertEqual(diff["device_cache.size"], expected_device_cache)
    else:
      # FIXME(feyu): Update these when the leaks are fixed.
      if size >= 20:
        self.assertEqual(diff["function_manager.size"], expected_cache_diff)
        self.assertEqual(
            diff["eager_pure_optimization.hit"], expected_eager_pure_hit
        )
        # TFRT doesn't use eager cache.
        if not test_util.is_tfrt_enabled():
          self.assertEqual(diff["kernel_cache.size"], expected_kernel_cache)
          self.assertEqual(diff["device_cache.size"], expected_device_cache)
      else:
        self.assertEqual(diff["function_manager.size"], expected_cache_diff)
        self.assertEqual(
            diff["eager_pure_optimization.hit"], expected_eager_pure_hit
        )
        # TFRT doesn't use eager cache.
        if not test_util.is_tfrt_enabled():
          self.assertEqual(diff["kernel_cache.size"], expected_kernel_cache)
          self.assertEqual(diff["device_cache.size"], expected_device_cache)

  def testManyEagerOpsVaryInput(self):
    c_layout = Layout.replicated(self.mesh, rank=10)

    c0 = constant_op.constant(
        [[[[[[[[[[0, 1, 2, 3], [4, 5, 6, 7]]]]]]]]]], dtype=dtypes.float32
    )
    e0 = c0.numpy()
    c0 = api.copy_to_mesh(c0, c_layout)

    for ax in range(10):
      self.assertAllEqual(
          math_ops.reduce_sum(c0, axis=ax).numpy(), np.sum(e0, axis=ax)
      )


if __name__ == "__main__":
  test.main()
