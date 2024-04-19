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

"""Tests for the internal DTensor Python API."""

from absl.testing import parameterized
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_random
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test

Layout = layout_lib.Layout
Mesh = layout_lib.Mesh
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'


class APITest(test_util.DTensorBaseTest):

  def setUp(self):
    super(APITest, self).setUp()
    global_ids = test_util.create_device_ids_array((2, 2))
    local_device_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        'CPU': Mesh(
            [_MESH_DIM_X, _MESH_DIM_Y],
            global_ids,
            local_device_ids,
            test_util.create_device_list((2, 2), 'CPU'),
        )
    }
    self.mesh = self.configTestMesh(mesh_dict)
    self.layouts_1d = [
        Layout.replicated(self.mesh, rank=1),
        Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=1),
        Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1),
    ]
    self.layouts_2d = [
        Layout.replicated(self.mesh, rank=2),
        Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=2),
        Layout.inner_sharded(self.mesh, _MESH_DIM_X, rank=2),
        Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh),
    ]

  def testV2API(self):
    layout = Layout.replicated(self.mesh, rank=1)
    zero_tensor = array_ops.zeros([10], layout=layout)
    zero_like_tensor = array_ops.zeros_like_v2(zero_tensor, layout=layout)
    self.assertAllEqual(zero_like_tensor.numpy(), zero_tensor.numpy())

    ones_tensor = array_ops.ones([10], layout=layout)
    ones_like_tensor = array_ops.ones_like_v2(zero_tensor, layout=layout)
    self.assertAllEqual(ones_like_tensor.numpy(), ones_tensor.numpy())

  def testStatelessRandom(self):
    # test dtype default float32 random
    result = stateless_random_ops.stateless_random_uniform(
        [10],
        seed=constant_op.constant([0, 0], dtype=dtypes.int64),
        minval=0.0,
        maxval=10.0,
    )
    self.assertEqual([10], result.shape)

    # test dtype default int32 minval maxval are both None
    result = stateless_random_ops.stateless_random_uniform(
        [10],
        seed=constant_op.constant([1, 2], dtype=dtypes.int64),
        dtype=dtypes.int32,
        minval=None,
        maxval=None,
    )
    self.assertEqual([10], result.shape)

    # test maxval is None or not given
    result = stateless_random_ops.stateless_random_uniform(
        [10],
        seed=constant_op.constant([1, 2], dtype=dtypes.int64),
        maxval=12,
        dtype=dtypes.int32,
    )
    self.assertEqual([10], result.shape)
    self.assertAllInRange(result, 0, 12)

  def testStatelessRandomNormal(self):
    # test dtype default float32 random
    result = stateless_random_ops.stateless_random_normal(
        [10], seed=constant_op.constant([0, 0], dtype=dtypes.int32)
    )
    self.assertEqual([10], result.shape)

    # test dtype double
    result = stateless_random_ops.stateless_random_normal(
        [10],
        seed=constant_op.constant([1, 2], dtype=dtypes.int32),
        dtype=dtypes.double,
    )
    self.assertEqual([10], result.shape)

    # test mean and stddev
    result = stateless_random_ops.stateless_random_normal(
        [10],
        seed=constant_op.constant([1, 2], dtype=dtypes.int32),
        mean=0,
        stddev=0,
    )
    self.assertEqual([10], result.shape)
    self.assertAllInRange(result, 0, 0)

    # test dtensor version of each, check layouts
    layout = Layout.replicated(self.mesh, rank=1)

    # test dtype default float 32 random
    result = d_random.stateless_random_normal(
        [10],
        seed=constant_op.constant([0, 0], dtype=dtypes.int32),
        layout=layout,
    )
    self.assertEqual([10], result.shape)
    self.assertEqual(layout, api.fetch_layout(result))

    # test dtype double
    result = d_random.stateless_random_normal(
        [10],
        seed=constant_op.constant([1, 2], dtype=dtypes.int32),
        dtype=dtypes.double,
        layout=layout,
    )
    self.assertEqual([10], result.shape)
    self.assertEqual(layout, api.fetch_layout(result))

    # test mean and stddev
    result = d_random.stateless_random_normal(
        [10],
        seed=constant_op.constant([1, 2], dtype=dtypes.int32),
        mean=0,
        stddev=0,
        layout=layout,
    )
    self.assertEqual([10], result.shape)
    self.assertAllInRange(result, 0, 0)
    self.assertEqual(layout, api.fetch_layout(result))

  @parameterized.named_parameters(*set(
      test_util.product((('_labels_unsharded', 0), ('_labels_batch', 1),
                         ('_labels_inner', 2), ('_labels_both', 3)),
                        (('_logits_unsharded', 0), ('_logits_batch', 1),
                         ('_logits_inner', 2), ('_logits_both', 3)))))
  def testSoftmaxCrossentropyWithLogits(self, labels_layout, logits_layout):
    expected_layout = Layout.replicated(self.mesh, rank=1)
    if (labels_layout == 1 or labels_layout == 3 or logits_layout == 1 or
        logits_layout == 3):
      expected_layout = Layout.inner_sharded(self.mesh, _MESH_DIM_X, rank=1)

    labels_layout = self.layouts_2d[labels_layout]
    logits_layout = self.layouts_2d[logits_layout]
    labels_numpy = np.random.uniform(size=[6, 4])
    logits_numpy = np.random.uniform(size=[6, 4])
    labels = constant_op.constant(labels_numpy, dtype=dtypes.float32)
    logits = constant_op.constant(logits_numpy, dtype=dtypes.float32)

    # Should we test against the built in version or the patched version?
    expected = nn_ops.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits
    )

    labels = numpy_util.pack_numpy(labels, labels_layout)
    logits = numpy_util.pack_numpy(logits, logits_layout)
    dtensor_result = nn_ops.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits
    )
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(*set(
      test_util.product((('_labels_unsharded', 0), ('_labels_batch_x', 1),
                         ('_labels_batch_y', 2)),
                        (('_logits_unsharded', 0), ('_logits_batch', 1),
                         ('_logits_inner', 2), ('_logits_both', 3)))))
  def testSparseSoftmaxCrossentropyWithLogits(self, labels_layout,
                                              logits_layout):
    expected_layout = Layout.replicated(self.mesh, rank=1)
    if labels_layout == 1 or logits_layout == 1 or logits_layout == 3:
      expected_layout = Layout.inner_sharded(self.mesh, _MESH_DIM_X, rank=1)
    elif labels_layout == 2:
      expected_layout = Layout.inner_sharded(self.mesh, _MESH_DIM_Y, rank=1)

    labels_layout = self.layouts_1d[labels_layout]
    logits_layout = self.layouts_2d[logits_layout]
    labels_numpy = np.random.randint(size=[6], low=0, high=4)
    logits_numpy = np.random.uniform(size=[6, 4])
    labels = constant_op.constant(labels_numpy, dtype=dtypes.int64)
    logits = constant_op.constant(logits_numpy, dtype=dtypes.float32)

    # Should we test against the built in version or the patched version?
    expected = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits
    )

    labels = numpy_util.pack_numpy(labels, labels_layout)
    logits = numpy_util.pack_numpy(logits, logits_layout)
    dtensor_result = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits
    )
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def test_dropout_raises_on_none_seed(self):
    with api.default_mesh(self.mesh):
      with self.assertRaisesRegex(ValueError, 'seed must be specified'):
        _ = d_random.dropout(
            array_ops.ones([2, 2], dtype=dtypes.float32), rate=0.5, seed=None
        )

  def test_default_mesh(self):

    @polymorphic_function.function
    def func(a):
      return a + 3.0

    with api.default_mesh(self.mesh):
      a = array_ops.zeros(shape=())
      result = func(a)

    self.assertEqual(result, 3.0)
    self.assertEqual(api.fetch_layout(result).mesh, self.mesh)
    self.assertTrue(api.fetch_layout(result).is_fully_replicated())
    self.assertEqual(result.device, api.device_name())

    # Also make sure it works as wrapper
    @api.default_mesh(self.mesh)
    def func2():
      b = array_ops.ones(shape=())
      return func(b)

    result = func2()
    self.assertEqual(result, 4.0)
    self.assertEqual(api.fetch_layout(result).mesh, self.mesh)
    self.assertTrue(api.fetch_layout(result).is_fully_replicated())
    self.assertEqual(result.device, api.device_name())

    with self.assertRaisesRegex(ValueError, 'Expect `mesh` to be `Mesh`'):
      with api.default_mesh(None):
        pass

  def test_default_mesh_with_constant(self):

    @polymorphic_function.function
    def func():
      return constant_op.constant([3, 4])

    with api.default_mesh(self.mesh):
      result = func()

    self.assertAllEqual(result, [3, 4])
    self.assertEqual(api.fetch_layout(result).mesh, self.mesh)
    self.assertTrue(api.fetch_layout(result).is_fully_replicated())
    self.assertEqual(result.device, api.device_name())

  def test_error_no_default_mesh(self):
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        'No default mesh has been registered to DTensor',
    ):
      with ops.device_v2(api.device_name()):
        _ = constant_op.constant(3.0)

  def test_get_default_mesh(self):
    self.assertIsNone(api.get_default_mesh())
    with api.default_mesh(self.mesh):
      self.assertEqual(api.get_default_mesh(), self.mesh)

      with api.default_mesh(self.mesh.host_mesh()):
        self.assertEqual(api.get_default_mesh(), self.mesh.host_mesh())

      self.assertEqual(api.get_default_mesh(), self.mesh)

    self.assertIsNone(api.get_default_mesh())


if __name__ == '__main__':
  test.main()
