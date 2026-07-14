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

"""Tests for batchparallel_spmd."""

import itertools
from absl.testing import parameterized
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.dtensor.python.tests import test_util_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
# pylint: enable=g-direct-tensorflow-import

Layout = layout_lib.Layout
Mesh = layout_lib.Mesh


class DTensorBatchParallelSPMDTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorBatchParallelSPMDTest, self).setUp()

    self.skipForDeviceType(['TPU'],
                           'all tests require 8 TPU cores.',
                           unless_device_count_equals_to=8)
    # Builds a 8x2 mesh.
    self._mesh_dim_b = 'b'
    self._mesh_dim_x = 'x'
    self._dims = [self._mesh_dim_b, self._mesh_dim_x]

    global_ids = test_util.create_device_ids_array((4, 2))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: Mesh(
            self._dims,
            global_ids,
            local_ids,
            test_util.create_device_list((4, 2), device),
        )
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)
    context.ensure_initialized()

    # Creates a bunch of common layouts used by tests later.
    # 4-d
    self.replicated_layout_4d = Layout.replicated(self.mesh, rank=4)
    self.batch_layout_4d = Layout.batch_sharded(
        self.mesh, self._mesh_dim_b, rank=4)

    # 5-d
    self.replicated_layout_5d = Layout.replicated(self.mesh, rank=5)
    self.batch_layout_5d = Layout.batch_sharded(
        self.mesh, self._mesh_dim_b, rank=5)

  @parameterized.named_parameters(('NoBatchDim', 0), ('SingleBatchDim', 1),
                                  ('TwoBatchDim', 2))
  def testCholesky(self, num_batch_dim):
    # Input needs to be symmetric and positive definite.
    x = constant_op.constant(
        [[1, 1, 1, 1], [1, 5, 5, 5], [1, 5, 14, 14], [1, 5, 14, 17]],
        dtype=dtypes.float32,
    )
    for _ in range(num_batch_dim):
      x = array_ops.expand_dims_v2(x, 0)
      s = [4] + [1 for _ in range(array_ops.rank(x) - 1)]
      x = gen_array_ops.tile(x, s)

    expected_result = gen_linalg_ops.cholesky(x)

    if num_batch_dim == 0:
      layout_spec = []
    elif num_batch_dim == 1:
      layout_spec = [self._mesh_dim_b]
    elif num_batch_dim == 2:
      layout_spec = [self._mesh_dim_b, self._mesh_dim_x]
    layout = Layout(layout_spec + ['unsharded'] * 2, self.mesh)

    x = numpy_util.pack_numpy(x, layout)
    got = gen_linalg_ops.cholesky(input=x)
    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(
      test_util.product(
          [('NoBatchDim', 0), ('SingleBatchDim', 1), ('TwoBatchDim', 2)],
          test_util_ops.FFT_OPS,
      )
  )
  def testFFT(self, num_batch_dim, fft_op, num_nonbatch_dim):
    shape = [4 for i in range(num_batch_dim + num_nonbatch_dim)]
    np.random.seed(123)
    x = constant_op.constant(
        np.random.normal(0.0, 1.0, np.prod(shape)).reshape(shape),
        dtype=dtypes.complex64,
    )
    expected_result = fft_op(input=x)

    if num_batch_dim == 0:
      layout_spec = []
    elif num_batch_dim == 1:
      layout_spec = [self._mesh_dim_b]
    elif num_batch_dim == 2:
      layout_spec = [self._mesh_dim_b, self._mesh_dim_x]
    layout = Layout(layout_spec + ['unsharded'] * num_nonbatch_dim, self.mesh)

    x = numpy_util.pack_numpy(x, layout)
    got = fft_op(input=x)
    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(
      test_util.product(
          [('NoBatchDim', 0), ('SingleBatchDim', 1), ('TwoBatchDim', 2)],
          test_util_ops.RFFT_OPS,
      )
  )
  def testRFFT(self, num_batch_dim, rfft_op, num_nonbatch_dim, dtype):
    self.skipForDeviceType(['GPU'], 'RFFT has numerical issues on GPU')
    shape = [4 for i in range(num_batch_dim + num_nonbatch_dim)]
    np.random.seed(123)
    x = constant_op.constant(
        np.random.normal(0.0, 1.0, np.prod(shape)).reshape(shape), dtype=dtype
    )
    expected_result = rfft_op(input=x, fft_length=[2] * num_nonbatch_dim)

    if num_batch_dim == 0:
      layout_spec = []
    elif num_batch_dim == 1:
      layout_spec = [self._mesh_dim_b]
    elif num_batch_dim == 2:
      layout_spec = [self._mesh_dim_b, self._mesh_dim_x]
    layout = Layout(layout_spec + ['unsharded'] * num_nonbatch_dim, self.mesh)

    x = numpy_util.pack_numpy(x, layout)
    got = rfft_op(input=x, fft_length=[2] * num_nonbatch_dim)
    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(
      test_util.product(
          [('Replicated', 'replicated'), ('Sharded', 'batch')],
          [
              (
                  'SamePadding',
                  'SAME',
              ),
              (
                  'ValidPadding',
                  'VALID',
              ),
          ],
          test_util_ops.BATCH_PARALLEL_2D_WINDOW_OPS,
      )
  )
  def test2DWindowOp(self, layout_spec, padding, op):
    np.random.seed(123)
    row_window_size = 3
    col_window_size = 4
    window_size = [1, row_window_size, col_window_size, 1]
    stride_size = [1, row_window_size - 1, col_window_size - 1, 1]

    num_rows = (row_window_size - 1) * 5 + 1
    num_cols = (col_window_size - 1) * 7 + 1
    x_in = np.random.normal(0.0, 1.0, 8 * num_rows * num_cols * 3).reshape(
        [8, num_rows, num_cols, 3])

    inputs = constant_op.constant(x_in, dtype=dtypes.float32)
    expected_result = op(inputs, window_size, stride_size, padding)

    if layout_spec == 'replicated':
      layout = self.replicated_layout_4d
    else:
      layout = self.batch_layout_4d

    x = numpy_util.pack_numpy(inputs, layout)
    got = op(x, window_size, stride_size, padding)
    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(
      test_util.product(
          [('Replicated', 'replicated'), ('BatchSharded', 'batch')],
          [
              (
                  'SamePadding',
                  'SAME',
              ),
              (
                  'ValidPadding',
                  'VALID',
              ),
          ],
          test_util_ops.BATCH_PARALLEL_3D_WINDOW_OPS,
      )
  )
  def test3DWindowOp(self, layout_spec, padding, op):
    np.random.seed(123)
    dep_window_size = 2
    row_window_size = 3
    col_window_size = 4
    window_size = [1, dep_window_size, row_window_size, col_window_size, 1]
    stride_size = [
        1, dep_window_size - 1, row_window_size - 1, col_window_size - 1, 1
    ]

    num_deps = 3
    num_rows = (row_window_size - 1) * 5 + 1
    num_cols = (col_window_size - 1) * 7 + 1
    x_in = np.random.normal(0.0, 1.0, 8 * num_deps * num_rows * num_cols *
                            3).reshape([8, num_deps, num_rows, num_cols, 3])

    inputs = constant_op.constant(x_in, dtype=dtypes.float32)
    expected_result = op(inputs, window_size, stride_size, padding)

    if layout_spec == 'replicated':
      layout = self.replicated_layout_5d
    else:
      layout = self.batch_layout_5d

    x = numpy_util.pack_numpy(inputs, layout)

    got = op(x, window_size, stride_size, padding)

    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(test_util_ops.PADDINGS)
  def testDepthwiseConv2dNative(self, padding):
    np.random.seed(123)
    x_in = np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1])

    kernel_in = np.array([
        [[[2, 0.1]], [[3, 0.2]]],
        [[[0, 0.3]], [[1, 0.4]]],
    ])

    inputs = constant_op.constant(x_in, dtype=dtypes.float32)
    kernel = constant_op.constant(kernel_in, dtype=dtypes.float32)
    expected_result = nn_impl.depthwise_conv2d_v2(
        inputs, kernel, strides=[1, 1, 1, 1], padding=padding
    )

    layout = self.batch_layout_4d

    x = numpy_util.pack_numpy(inputs, layout)
    kernel = numpy_util.pack_numpy(kernel, self.replicated_layout_4d)
    got = nn_impl.depthwise_conv2d_v2(
        x, kernel, strides=[1, 1, 1, 1], padding=padding
    )

    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(('Sharded', 'sharded'),
                                  ('Replicated', 'replicated'))
  def testResizeBilinear(self, shard_spec):
    np.random.seed(123)
    images = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1]),
        dtype=dtypes.float32,
    )

    expected_result = gen_image_ops.resize_bilinear(
        images=images,
        size=[3, 3],
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )

    if shard_spec == 'sharded':
      layout = self.batch_layout_4d
    else:
      layout = self.replicated_layout_4d
    images = numpy_util.pack_numpy(images, layout)

    got = gen_image_ops.resize_bilinear(
        images=images,
        size=[3, 3],
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )

    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(('Sharded', 'sharded'),
                                  ('Replicated', 'replicated'))
  def testResizeNearestNeighbor(self, shard_spec):
    np.random.seed(123)
    images = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1]),
        dtype=dtypes.float32,
    )

    expected_result = gen_image_ops.resize_nearest_neighbor(
        images=images,
        size=[3, 3],
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )

    if shard_spec == 'sharded':
      layout = self.batch_layout_4d
    else:
      layout = self.replicated_layout_4d
    images = numpy_util.pack_numpy(images, layout)

    got = gen_image_ops.resize_nearest_neighbor(
        images=images,
        size=[3, 3],
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )

    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(('Sharded', 'sharded'),
                                  ('Replicated', 'replicated'))
  def testAdjustContrastv2(self, shard_spec):
    np.random.seed(123)
    images = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9 * 3).reshape([8, 9, 9, 3]),
        dtype=dtypes.float32,
    )

    expected_result = gen_image_ops.adjust_contrastv2(
        images=images, contrast_factor=0.5
    )

    if shard_spec == 'sharded':
      layout = self.batch_layout_4d
    else:
      layout = self.replicated_layout_4d
    images = numpy_util.pack_numpy(images, layout)

    got = gen_image_ops.adjust_contrastv2(images=images, contrast_factor=0.5)

    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(('Sharded', 'sharded'),
                                  ('Replicated', 'replicated'))
  def testAdjustSaturation(self, shard_spec):
    np.random.seed(123)
    images = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9 * 3).reshape([8, 9, 9, 3]),
        dtype=dtypes.float32,
    )

    expected_result = gen_image_ops.adjust_saturation(images=images, scale=0.5)

    if shard_spec == 'sharded':
      layout = self.batch_layout_4d
    else:
      layout = self.replicated_layout_4d
    images = numpy_util.pack_numpy(images, layout)

    got = gen_image_ops.adjust_saturation(images=images, scale=0.5)

    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.parameters(
      itertools.permutations(['sharded', 'replicated'], 2))
  def testResizeBilinearGradBatchSharded(self, spec1, spec2):
    np.random.seed(123)
    images = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1]),
        dtype=dtypes.float32,
    )
    grads = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1]),
        dtype=dtypes.float32,
    )
    expected_result = gen_image_ops.resize_bilinear_grad(
        grads=grads,
        original_image=images,
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )

    specs = [spec1, spec2]
    layouts = [
        self.batch_layout_4d if spec == 'sharded' else self.replicated_layout_4d
        for spec in specs
    ]

    # Test images is replicated, grads is batch sharded
    images = numpy_util.pack_numpy(images, layouts[0])
    grads = numpy_util.pack_numpy(grads, layouts[1])

    got = gen_image_ops.resize_bilinear_grad(
        grads=grads,
        original_image=images,
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )
    self.assertDTensorEqual(expected_result, self.batch_layout_4d, got)

  def testResizeBilinearGradReplicated(self):
    np.random.seed(123)
    images = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1]),
        dtype=dtypes.float32,
    )
    grads = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1]),
        dtype=dtypes.float32,
    )
    expected_result = gen_image_ops.resize_bilinear_grad(
        grads=grads,
        original_image=images,
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )

    images = numpy_util.pack_numpy(images, self.replicated_layout_4d)
    grads = numpy_util.pack_numpy(grads, self.replicated_layout_4d)

    got = gen_image_ops.resize_bilinear_grad(
        grads=grads,
        original_image=images,
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )
    self.assertDTensorEqual(expected_result, self.replicated_layout_4d, got)

  @parameterized.named_parameters(
      test_util.product([('Replicated', 'replicated'), ('Sharded', 'batch')], [(
          'SamePadding',
          'SAME',
      ), (
          'ValidPadding',
          'VALID',
      )]))
  def testMaxPool3DGrad(self, shard_spec, padding):
    np.random.seed(123)
    dep_window_size = 2
    row_window_size = 3
    col_window_size = 4
    window_size = [1, dep_window_size, row_window_size, col_window_size, 1]
    stride_size = [
        1, dep_window_size - 1, row_window_size - 1, col_window_size - 1, 1
    ]

    num_deps = 3
    num_rows = (row_window_size - 1) * 5 + 1
    num_cols = (col_window_size - 1) * 7 + 1
    x_in = np.random.normal(0.0, 1.0, 8 * num_deps * num_rows * num_cols *
                            3).reshape([8, num_deps, num_rows, num_cols, 3])
    inputs = constant_op.constant(x_in, dtype=dtypes.float32)

    with backprop.GradientTape() as tape:
      tape.watch([inputs])
      expected_result = nn_ops.max_pool3d(
          inputs, window_size, stride_size, padding
      )
    expected_grad = tape.gradient(expected_result, [inputs])
    layout = (
        self.batch_layout_5d
        if shard_spec == 'sharded'
        else self.replicated_layout_5d
    )

    inputs = numpy_util.pack_numpy(inputs, layout)

    with ops.device_v2(api.device_name()):
      with backprop.GradientTape() as tape:
        tape.watch([inputs])
        dtensor_result = nn_ops.max_pool3d(
            inputs, window_size, stride_size, padding
        )
      dtensor_grad = tape.gradient(dtensor_result, [inputs])

    self.assertDTensorEqual(expected_grad[0], layout, dtensor_grad[0])

  @parameterized.named_parameters(
      test_util.product([('Replicated', 'replicated'), ('Sharded', 'batch')], [(
          'SamePadding',
          'SAME',
      ), (
          'ValidPadding',
          'VALID',
      )]))
  def testMaxPool3DGradGrad(self, shard_spec, padding):
    np.random.seed(123)
    dep_window_size = 2
    row_window_size = 3
    col_window_size = 4
    window_size = [1, dep_window_size, row_window_size, col_window_size, 1]
    stride_size = [
        1, dep_window_size - 1, row_window_size - 1, col_window_size - 1, 1
    ]

    num_deps = 3
    num_rows = (row_window_size - 1) * 5 + 1
    num_cols = (col_window_size - 1) * 7 + 1
    x_in = np.random.normal(0.0, 1.0, 8 * num_deps * num_rows * num_cols *
                            3).reshape([8, num_deps, num_rows, num_cols, 3])
    inputs = constant_op.constant(x_in, dtype=dtypes.float32)

    with backprop.GradientTape() as outer_tape:
      with backprop.GradientTape() as inner_tape:
        outer_tape.watch([inputs])
        inner_tape.watch([inputs])
        expected_result = nn_ops.max_pool3d(
            inputs, window_size, stride_size, padding
        )
      expected_first_grad = inner_tape.gradient(expected_result, [inputs])
    expected_second_grad = outer_tape.gradient(expected_first_grad, [inputs])

    if shard_spec == 'sharded':
      layout = self.batch_layout_5d
    else:
      layout = self.replicated_layout_5d

    inputs = numpy_util.pack_numpy(inputs, layout)

    @polymorphic_function.function()
    def compute_gradients(inputs):
      with backprop.GradientTape() as outer_tape:
        with backprop.GradientTape() as inner_tape:
          outer_tape.watch([inputs])
          inner_tape.watch([inputs])
          dtensor_result = nn_ops.max_pool3d(
              inputs, window_size, stride_size, padding
          )
        dtensor_first_grad = inner_tape.gradient(dtensor_result, [inputs])
      dtensor_second_grad = outer_tape.gradient(dtensor_first_grad[0], [inputs])
      return dtensor_first_grad, dtensor_second_grad

    dtensor_first_grad, dtensor_second_grad = compute_gradients(inputs)

    self.assertDTensorEqual(expected_first_grad[0], layout,
                            dtensor_first_grad[0])
    self.assertDTensorEqual(expected_second_grad[0], layout,
                            dtensor_second_grad[0])

  @parameterized.named_parameters(
      test_util.product([('Replicated', 'replicated'), ('Sharded', 'batch')], [(
          'SamePadding',
          'SAME',
      ), (
          'ValidPadding',
          'VALID',
      )]))
  def testMaxPoolGradGrad(self, shard_spec, padding):
    np.random.seed(123)
    row_window_size = 3
    col_window_size = 4
    window_size = [1, row_window_size, col_window_size, 1]
    stride_size = [1, row_window_size - 1, col_window_size - 1, 1]

    num_rows = (row_window_size - 1) * 5 + 1
    num_cols = (col_window_size - 1) * 7 + 1
    x_in = np.random.normal(0.0, 1.0, 8 * num_rows * num_cols * 3).reshape(
        [8, num_rows, num_cols, 3])
    inputs = constant_op.constant(x_in, dtype=dtypes.float32)

    with backprop.GradientTape() as outer_tape:
      with backprop.GradientTape() as inner_tape:
        outer_tape.watch([inputs])
        inner_tape.watch([inputs])
        expected_result = nn_ops.max_pool_v2(
            inputs, window_size, stride_size, padding
        )
      expected_first_grad = inner_tape.gradient(expected_result, [inputs])
    expected_second_grad = outer_tape.gradient(expected_first_grad, [inputs])

    if shard_spec == 'sharded':
      layout = self.batch_layout_4d
    else:
      layout = self.replicated_layout_4d
    inputs = numpy_util.pack_numpy(inputs, layout)

    @polymorphic_function.function()
    def compute_gradients(inputs):
      with backprop.GradientTape() as outer_tape:
        with backprop.GradientTape() as inner_tape:
          outer_tape.watch([inputs])
          inner_tape.watch([inputs])
          dtensor_result = nn_ops.max_pool_v2(
              inputs, window_size, stride_size, padding
          )
        dtensor_first_grad = inner_tape.gradient(dtensor_result, [inputs])
      dtensor_second_grad = outer_tape.gradient(dtensor_first_grad[0], [inputs])
      return dtensor_first_grad, dtensor_second_grad

    dtensor_first_grad, dtensor_second_grad = compute_gradients(inputs)

    self.assertDTensorEqual(expected_first_grad[0], layout,
                            dtensor_first_grad[0])
    self.assertDTensorEqual(expected_second_grad[0], layout,
                            dtensor_second_grad[0])

  @parameterized.named_parameters(('Sharded', 'sharded'),
                                  ('Replicated', 'replicated'))
  def testResizeNearestNeighborGrad(self, shard_spec):
    np.random.seed(123)
    grads = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1]),
        dtype=dtypes.float32,
    )
    expected_result = gen_image_ops.resize_nearest_neighbor_grad(
        grads=grads,
        size=[3, 3],
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )

    if shard_spec == 'sharded':
      layout = self.batch_layout_4d
    else:
      layout = self.replicated_layout_4d

    grads = numpy_util.pack_numpy(grads, layout)

    got = gen_image_ops.resize_nearest_neighbor_grad(
        grads=grads,
        size=[3, 3],
        align_corners=False,
        half_pixel_centers=False,
        name=None,
    )

    self.assertDTensorEqual(expected_result, layout, got)


if __name__ == '__main__':
  test.main()
