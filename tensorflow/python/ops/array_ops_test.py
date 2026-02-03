# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for array operations."""

from tensorflow.core.config import flags
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.framework import config
from tensorflow.python.framework import random_seed


class ArrayOpTest(test.TestCase):

  def testGatherGradHasPartialStaticShape(self):
    # Create a tensor with an unknown dim 1.
    x = random_ops.random_normal([4, 10, 10])
    x = array_ops.gather(
        x, array_ops.reshape(array_ops.where_v2(x[0, :, 0] > 0.5), [-1]), axis=1
    )
    x.shape.assert_is_compatible_with([4, None, 10])

    with backprop.GradientTape() as tape:
      tape.watch(x)
      a = array_ops.gather(array_ops.gather(x, [0, 1]), [0, 1])
    grad_a = tape.gradient(a, x)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      b = array_ops.gather(array_ops.gather(x, [2, 3], axis=2), [0, 1])
    grad_b = tape.gradient(b, x)

    # We make sure that the representation of the shapes are correct; the shape
    # equality check will always eval to false due to the shapes being partial.
    grad_a.shape.assert_is_compatible_with([None, None, 10])
    grad_b.shape.assert_is_compatible_with([4, None, 10])

  def testReshapeShapeInference(self):
    # Create a tensor with an unknown dim 1.
    x = random_ops.random_normal([4, 10, 10])
    x = array_ops.gather(
        x, array_ops.reshape(array_ops.where_v2(x[0, :, 0] > 0.5), [-1]), axis=1
    )
    x.shape.assert_is_compatible_with([4, None, 10])
    a = array_ops.reshape(x, array_ops.shape(x))
    a.shape.assert_is_compatible_with([4, None, 10])
    b = array_ops.reshape(x, math_ops.cast(array_ops.shape(x), dtypes.int64))
    b.shape.assert_is_compatible_with([4, None, 10])

    # We do not shape-infer across a tf.cast into anything that's not tf.int32
    # or tf.int64, since they might end up mangling the shape.
    c = array_ops.reshape(
        x,
        math_ops.cast(
            math_ops.cast(array_ops.shape(x), dtypes.float32), dtypes.int32
        ),
    )
    c.shape.assert_is_compatible_with([None, None, None])

  def testEmptyMeshgrid(self):
    self.assertEqual(array_ops.meshgrid(), [])

  def testSlicedPartialShapeInference(self):
    @def_function.function(autograph=False)
    def g(x):
      return array_ops.zeros([array_ops.shape(x)[0]])

    conc = g.get_concrete_function(tensor_spec.TensorSpec([10, None]))
    self.assertAllEqual(conc.output_shapes.as_list(), [10])

  def testIdentityOnSlicedPartialShapeInference(self):
    @def_function.function(autograph=False)
    def g(x):
      return array_ops.zeros([array_ops.identity(array_ops.shape(x)[0])])

    conc = g.get_concrete_function(tensor_spec.TensorSpec([10, None]))
    self.assertAllEqual(conc.output_shapes.as_list(), [10])

  @test_util.run_in_graph_and_eager_modes
  def testParallelConcatFailsWithRankZeroShape(self):
    op = array_ops.ParallelConcat
    para = {"shape": 0, "values": [1]}

    def func():
      y = op(**para)
      return y

    with self.assertRaisesRegex(
        Exception, "(rank|dimension) of .* must be greater than .* 0"
    ):
      func()

  @test_util.run_in_graph_and_eager_modes
  def testUpperBoundValuesWrongRank(self):
    # Used to cause a segfault, b/266336058
    arg0 = array_ops.zeros([2, 3], dtype=dtypes.float32)
    arg1 = array_ops.zeros([2, 1, 0], dtype=dtypes.float32)
    with self.assertRaisesRegex(
        Exception, "Shape must be rank 2 but is rank 3"
    ):
      gen_array_ops.upper_bound(arg0, arg1)

  def testLowerBoundValuesWrongRank(self):
    # Used to cause a segfault, b/266336058
    arg0 = array_ops.zeros([2, 3], dtype=dtypes.float32)
    arg1 = array_ops.zeros([2, 1, 0], dtype=dtypes.float32)
    with self.assertRaisesRegex(
        Exception, "Shape must be rank 2 but is rank 3"
    ):
      gen_array_ops.lower_bound(arg0, arg1)

  def testUpperBoundInputsWrongRank(self):
    # Used to cause a segfault, b/266336058
    arg0 = array_ops.zeros([2, 1, 0], dtype=dtypes.float32)
    arg1 = array_ops.zeros([2, 3], dtype=dtypes.float32)
    with self.assertRaisesRegex(
        Exception, "Shape must be rank 2 but is rank 3"
    ):
      gen_array_ops.upper_bound(arg0, arg1)

  def testLowerBoundInputsWrongRank(self):
    # Used to cause a segfault, b/266336058
    arg0 = array_ops.zeros([2, 1, 0], dtype=dtypes.float32)
    arg1 = array_ops.zeros([2, 3], dtype=dtypes.float32)
    with self.assertRaisesRegex(
        Exception, "Shape must be rank 2 but is rank 3"
    ):
      gen_array_ops.lower_bound(arg0, arg1)

  def testShapeDefaultIn32(self):
    # The tf_shape_default_int64 flag should NOT be set when this test runs
    self.assertFalse(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.shape_v2(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int32)

class TestFoldNonOverlapping(test.TestCase):

  def setUp(self):
      super().setUp()
      random_seed.set_seed(42)

  def _extract_patches(self,x, kernel, stride, padding="VALID", dilation=1):
    """Helper that matches TensorFlow's _extract_patches API"""
    return array_ops.extract_image_patches_v2(
        images=x,
        sizes=[1, kernel, kernel, 1],
        strides=[1, stride, stride, 1],
        rates=[1, dilation, dilation, 1],
        padding=padding,
    )

  def test_perfect_inverse_no_overlap_valid_basic(self):
    x = random_ops.random_normal([2,8,8,3])
    patches = self._extract_patches(x, kernel=4, stride=4, padding='VALID')
    reconstructed = array_ops.fold(
            patches,
            output_size=(8, 8),
            kernel_size=4,
            stride=4,
            padding='VALID'
        )
    self.assertAllClose(reconstructed,x,msg="fold() is not the perfect inverse of _extract_patches (VALID)")

  def test_inverse_various_sizes_no_overlap(self):
    """Test to see if inverse relationship holds for various batch, image, kernel, and channel sizes"""
    batch_sizes = [1, 2, 4]
    image_sizes = [6, 8, 12, 16]
    kernel_sizes = [2, 3, 4]
    channel_sizes = [1, 3, 4, 8]

    for batch_size in batch_sizes:
        for image_size in image_sizes:
            for kernel_size in kernel_sizes:
                for channels in channel_sizes:
                    if image_size % kernel_size != 0:
                        continue

                    x = random_ops.random_normal([batch_size, image_size, image_size, channels])
                    patches = self._extract_patches(x, kernel=kernel_size, stride=kernel_size, padding='VALID')
                    reconstructed = array_ops.fold(
                        patches,
                        output_size=(image_size, image_size),
                        kernel_size=kernel_size,
                        stride=kernel_size,
                        padding='VALID'
                    )

                    self.assertAllClose(reconstructed, x)
  
  def test_dilation_parameter_compatibility(self):
    x = array_ops.reshape(math_ops.range(0, 16, dtype=dtypes.float32), (1, 4, 4, 1))

    dilations = [1, 2]
    for dilation in dilations:
        patches = self._extract_patches(x, kernel=2, stride=2, padding='VALID', dilation=dilation)

        out = array_ops.fold(
            patches,
            output_size=(4, 4),
            kernel_size=2,
            stride=2,
            padding='VALID',
            dilation=dilation
        )
        self.assertAllEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        # Output should not be all zeros or NaN
        # self.assertFalse(math_ops.reduce_all(math_ops.equal(out, 0)).numpy())
        # self.assertFalse(math_ops.reduce_any(math_ops.is_nan(out)).numpy())


class TestFoldOverlapping(test.TestCase):
  
  def _extract_patches(self,x, kernel, stride, padding="VALID", dilation=1):
    """Helper that matches TensorFlow's _extract_patches API"""
    return array_ops.extract_image_patches_v2(
        images=x,
        sizes=[1, kernel, kernel, 1],
        strides=[1, stride, stride, 1],
        rates=[1, dilation, dilation, 1],
        padding=padding,
    )
  
  def setUp(self):
        super().setUp()
        random_seed.set_seed(42)
        config.enable_op_determinism()

  def test_fold_overlapping_patches_basic(self):
    # stride < kernel for overlap,
    # -> (image_size - kernel) must be divisible by stride
    x = array_ops.reshape(math_ops.range(0, 16, dtype=dtypes.float32), (1, 4, 4, 1))
    patches = self._extract_patches(x, kernel=2, stride=1, padding='VALID')
    reconstructed = array_ops.fold(
            patches,
            output_size=(4, 4),
            kernel_size=2,
            stride=1,
            padding='VALID'
        )
    
    self.assertAllEqual(reconstructed.shape, x.shape)
    # print(f"original: {x} \n\n Unfold: {patches} \n\n Fold:{reconstructed}")
    self.assertEqual(reconstructed.dtype, x.dtype)
    
    overlap_counts = array_ops.constant([ #manual calc
        [[1], [2], [2], [1]],
        [[2], [4], [4], [2]],
        [[2], [4], [4], [2]],
        [[1], [2], [2], [1]],
    ], dtype=dtypes.float32)
    overlap_counts = array_ops.reshape(overlap_counts, (1, 4, 4, 1))

    expected = x * overlap_counts

    self.assertAllClose(reconstructed, expected)

  def test_fold_overlapping_patches_various_params(self):
    """Test overlapping fold with VALID padding across different kernel/stride combos"""
    batch_sizes = [1, 2]
    channel_sizes = [1, 3]
    params = [
        (6, 4, 2),   
        (6, 3, 1),   
        (8, 4, 2),   
        (8, 6, 2),   
    ]

    for batch_size in batch_sizes:
        for channels in channel_sizes:
            for image_size, kernel_size, stride in params:
                x = random_ops.random_normal([batch_size, image_size, image_size, channels])
                patches = self._extract_patches(x, kernel=kernel_size, stride=stride, padding='VALID')

                reconstructed = array_ops.fold(
                    patches,
                    output_size=(image_size, image_size),
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='VALID'
                )
                # Building the overlap count map by folding a tensor of ones.
                # Each position accumulates how many patches it belongs to.
                ones = array_ops.ones_like(x)
                ones_patches = self._extract_patches(ones, kernel=kernel_size, stride=stride, padding='VALID')
                overlap_counts = array_ops.fold(
                    ones_patches,
                    output_size=(image_size, image_size),
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='VALID'
                )
                # print(f"Original: {x} \n\n Overlap_count:{overlap_counts}")
                # print(f"Args: \n Batch: {batch_size}; Channels:{channels}; Image_size:{image_size}; Kernel_size:{kernel_size}; Stride:{stride}")
                expected = x * overlap_counts                
                self.assertAllEqual(reconstructed.shape, x.shape)      
                self.assertEqual(reconstructed.dtype, x.dtype)
                self.assertAllClose(reconstructed, expected)



if __name__ == "__main__":
  test.main()
