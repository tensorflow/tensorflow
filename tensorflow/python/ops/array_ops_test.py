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
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.framework import config
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import gradient_checker_v2 


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

@test_util.run_all_in_graph_and_eager_modes
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
            sizes=4,
            strides=4,
            padding='VALID'
        )
    self.assertAllClose(
      reconstructed,x,
      msg="fold() is not the perfect inverse of _extract_patches (VALID)")

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
            x = random_ops.random_normal(
              [batch_size, image_size, image_size, channels])
            patches = self._extract_patches(
              x, kernel=kernel_size, 
              stride=kernel_size, padding='VALID')
            reconstructed = array_ops.fold(
              patches,
              output_size=(image_size, image_size),
              sizes=kernel_size,
              strides=kernel_size,
              padding='VALID'
              )
            self.assertAllClose(reconstructed, x)
  
  def test_dilation_parameter_compatibility(self):
    x = array_ops.reshape(
      math_ops.range(0, 16, dtype=dtypes.float32), (1, 4, 4, 1))

    dilations = [1, 2]
    for dilation in dilations:
      patches = self._extract_patches(
        x, kernel=2, stride=2, padding='VALID', dilation=dilation)

      out = array_ops.fold(
          patches,
          output_size=(4, 4),
          sizes=2,
          strides=2,
          padding='VALID',
          rates=dilation
      )
      self.assertAllEqual(out.shape, x.shape)
      self.assertEqual(out.dtype, x.dtype)


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
    x = array_ops.reshape(
      math_ops.range(0, 16, dtype=dtypes.float32), (1, 4, 4, 1))
    patches = self._extract_patches(x, kernel=2, stride=1, padding='VALID')
    reconstructed = array_ops.fold(
            patches,
            output_size=(4, 4),
            sizes=2,
            strides=1,
            padding='VALID'
        )
    
    self.assertAllEqual(reconstructed.shape, x.shape)
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
          x = random_ops.random_normal(
            [batch_size, image_size, image_size, channels])
          patches = self._extract_patches(
            x, kernel=kernel_size, stride=stride, padding='VALID')

          reconstructed = array_ops.fold(
              patches,
              output_size=(image_size, image_size),
              sizes=kernel_size,
              strides=stride,
              padding='VALID'
          )
          # Building the overlap count map by folding a tensor of ones.
          # Each position accumulates how many patches it belongs to.
          ones = array_ops.ones_like(x)
          ones_patches = self._extract_patches(
            ones, kernel=kernel_size, stride=stride, padding='VALID')
          overlap_counts = array_ops.fold(
              ones_patches,
              output_size=(image_size, image_size),
              sizes=kernel_size,
              strides=stride,
              padding='VALID'
          )              
          expected = x * overlap_counts                
          self.assertAllEqual(reconstructed.shape, x.shape)      
          self.assertEqual(reconstructed.dtype, x.dtype)
          self.assertAllClose(reconstructed, expected)

  @test_util.run_in_graph_and_eager_modes
  def testFoldReductionMeanReconstruction(self):
    """Basic test for reduction='mean' """
    input_image = array_ops.constant(
        [[[[1.0], [2.0], [3.0]],
          [[4.0], [5.0], [6.0]],
          [[7.0], [8.0], [9.0]]]]
    )

    # The center pixel (5.0) will be duplicated 4 times!
    patches = array_ops.extract_image_patches(
        input_image, 
        ksizes=[1, 2, 2, 1], 
        strides=[1, 1, 1, 1], 
        rates=[1, 1, 1, 1], 
        padding='VALID'
    )

    # patches = [[1, 2, 4, 5],
    #            [2, 3, 5, 6],
    #            [4, 5, 7, 8],
    #            [5, 6, 8, 9]

    # *(divisor_matrix) represents overlap count
    # Eg. element 5.0 (in the input image,center pixel) was overlapped 4 times!
    # [  1    2    1 ] 
    # [  2    4    2 ]
    # [  1    2    1 ]

    # folded_img (with reduction='sum')
    # [  1    4    3 ]  
    # [  8   20   12 ]
    # [  7   16    9 ]

    folded_image = array_ops.fold(
        patches,
        output_size=(3, 3),
        sizes=2,
        strides=1,
        rates=1,
        padding='VALID',
        reduction='mean'
    )

    #* folded_img (with reduction='mean') = folded_img_summed ⊘ overlap count (ones_tensor)
    # [  1/1    4/2    3/1 ]  = [1,2,3]
    # [  8/2   20/4   12/2 ]  = [4,5,6]
    # [  7/1   16/2    9/1 ]  = [7,8,9]

    # reduction='mean' should recreate the original image.
    self.assertAllClose(input_image, folded_image)

  @test_util.run_in_graph_and_eager_modes
  def testFoldReductionMeanEmptyPixels(self):
    """This tests the `safe_divisor` logic to ensure no NaNs are produced 
    when pixels receive 0 patches."""
    patches = array_ops.constant(
        [[[[1.0, 2.0, 3.0, 4.0]]]]
    ) # created a single 2x2 patch

    # fold it into a 3x3 output space.
    # The bottom and right edges will have 0 overlap
    folded_image = array_ops.fold(
        patches,
        output_size=(3, 3),
        sizes=2,
        strides=1,
        rates=1,
        padding='VALID',
        reduction='mean'
    )

    # 1. Ensuring NO NaNs exist in the output 
    has_nans = math_ops.reduce_any(math_ops.is_nan(folded_image))
    self.assertFalse(self.evaluate(has_nans))

    # 2. Ensuring the untouched pixels defaulted to 0.0
    # Check coordinate (row 0, col 2) which is outside the 2x2 patch
    self.assertEqual(self.evaluate(folded_image[0, 0, 2, 0]), 0.0)

  @test_util.run_in_graph_and_eager_modes
  def testFoldReductionMeanMultipleDimensions(self):
    """ We test combinations of (batch_size, channels, size, stride, rate)"""

    test_configs = [
        {"batch": 2, "channels": 3, "size": 2, "stride": 1, "rate": 2},
        {"batch": 3, "channels": 2, "size": 3, "stride": 2, "rate": 1},
    ]

    for config in test_configs:
      
      input_shape = [config["batch"], 7, 7, config["channels"]]
      input_image = random_ops.random_uniform(input_shape, seed=42)

      
      patches = array_ops.extract_image_patches(
          input_image,
          ksizes=[1, config["size"], config["size"], 1],
          strides=[1, config["stride"], config["stride"], 1],
          rates=[1, config["rate"], config["rate"], 1],
          padding='VALID'
      )

      
      folded_image = array_ops.fold(
          patches,
          output_size=(7, 7),
          sizes=config["size"],
          strides=config["stride"],
          rates=config["rate"],
          padding='VALID',
          reduction='mean'
      )

      
      self.assertAllClose(input_image, folded_image)

class TestFoldInputValidation(test.TestCase):
  """Also checks error handling"""

  def setUp(self):
    super().setUp()
    random_seed.set_seed(42)
    
  def test_invalid_dilation_raises(self):
    patches = random_ops.random_normal([1, 3, 3, 4])
    with self.assertRaisesRegex(ValueError, "dilation must be >= 1"):
      array_ops.fold(
        patches, output_size=(4, 4), sizes=2, strides=2, rates=-1)
  
  def test_invalid_padding_string_raises(self):
    patches = random_ops.random_normal([1, 3, 3, 4])
    
    with self.assertRaisesRegex(ValueError, "padding must be"):
      array_ops.fold(
        patches, 
        output_size=(4, 4), 
        sizes=2, 
        strides=2, 
        padding='INVALID_STRING')

  def test_invalid_image_input_size(self):
    patches = random_ops.random_normal([3, 3, 4])  
    with self.assertRaisesRegex(ValueError, "input must be 4D"):
      array_ops.fold(
        patches, output_size=(4, 4), sizes=2, strides=2)

  def test_patch_dim_not_divisible_by_kernel_raises(self):
    patches = random_ops.random_normal([1, 3, 3, 5]) 
    with self.assertRaisesRegex(
      ValueError,
      "input's dimension 3 should be divisble by the product of kernel_size"):        
      array_ops.fold(patches, output_size=(4, 4), sizes=2, strides=2)
  
  def test_invalid_kernel_size(self):
    patches = random_ops.random_normal([1, 4, 4, 1]) 
    with self.assertRaisesRegex(ValueError,"kernel_size must be >= 1"):
      array_ops.fold(patches, (4, 4),(-1,1),2)
      array_ops.fold(patches, (4, 4),(2,-1),2)
      array_ops.fold(patches, (4, 4),2,2)
  
  def test_invalid_stride(self):
    patches = random_ops.random_normal([1, 4, 4, 1]) 
    with self.assertRaisesRegex(ValueError,"stride must be >= 1"):
      array_ops.fold(patches, (4, 4),2,(-1,1))
      array_ops.fold(patches, (4, 4),2,(1,-2))
      array_ops.fold(patches, (4, 4),2,2)

@test_util.run_all_in_graph_and_eager_modes
class TestFoldGradients(test.TestCase):
  """Verifies that fold is differentiable and produces numerically correct gradients 
  when composed with extract_patches."""

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
  
  def test_fold_gradient_exists(self):
    """Verifies that fold produces non-zero gradients """
    x = random_ops.random_normal([1, 4, 4, 1]) 

    with backprop.GradientTape() as tape:
      tape.watch(x)

      patches = self._extract_patches(x,kernel=2,stride=1,padding="VALID")
      y = array_ops.fold(
        patches,
        output_size=(4, 4),
        sizes=2,
        strides=1,
        padding="VALID")
      loss = math_ops.reduce_sum(y)
    
    grad = tape.gradient(loss, x)

    self.assertIsNotNone(grad)
    self.assertGreater(
      self.evaluate(math_ops.reduce_sum(math_ops.abs(grad))), 0.0)
    
  def test_fold_gradient_numerical_correctness(self):
    """To check if autodiff matches numerical gradient """
    x = random_ops.random_normal([1, 4, 4, 1]) 

    def forward(x):
      patches = self._extract_patches(x,kernel=2,stride=1,padding="VALID")
      y = array_ops.fold(
        patches,
        output_size=(4, 4),
        sizes=2,
        strides=1,
        padding="VALID")
      return math_ops.reduce_sum(y)

    with self.cached_session():
      theoretical, numerical = gradient_checker_v2.compute_gradient(
        forward, [x])

    self.assertAllClose(
      theoretical[0],
      numerical[0],
      atol=1e-3,
      rtol=1e-3,
      msg="Autodiff and numerical gradients don't match",
      )
@test_util.run_all_in_graph_and_eager_modes 
class TestFoldSamePadding(test.TestCase):
  
  def _extract_patches(self, x, kernel, stride, padding="VALID", dilation=1):
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

  def test_perfect_inverse_no_overlap_same(self):
    """Verifies SAME padding reconstructs perfectly when stride == kernel.
    An image size of 5 with kernel/stride 3 forces 1 pixel of padding."""
    x = random_ops.random_normal([2, 5, 5, 3],seed=42)
    patches = self._extract_patches(x, kernel=3, stride=3, padding='SAME')
    
    reconstructed = array_ops.fold(
        patches,
        output_size=(5, 5),
        sizes=3,
        strides=3,
        padding='SAME'
    )
    
    self.assertAllClose(
        reconstructed, x,
        msg="fold() is not a perfect inverse with SAME padding"
    )

  def test_fold_asymmetric_same_padding(self):
    """Testing the (pad_total // 2) logic. 
    Image 6x6, Kernel 3, Stride 2 requires EXACTLY 1 pixel of total padding.
    TensorFlow puts this on the bottom/right."""
    image_size, kernel_size, stride = 6, 3, 2
    x = random_ops.random_normal([1, image_size, image_size, 1],seed=42)
    
    patches = self._extract_patches(
        x, kernel=kernel_size, stride=stride, padding='SAME')
    
    reconstructed = array_ops.fold(
        patches,
        output_size=(image_size, image_size),
        sizes=kernel_size,
        strides=stride,
        padding='SAME'
    )

    #* Use the ones-tensor to calculate expected overlap accumulation
    ones = array_ops.ones_like(x)
    ones_patches = self._extract_patches(
        ones, kernel=kernel_size, stride=stride, padding='SAME')
    overlap_counts = array_ops.fold(
        ones_patches,
        output_size=(image_size, image_size),
        sizes=kernel_size,
        strides=stride,
        padding='SAME'
    )
    
    expected = x * overlap_counts
    self.assertAllClose(reconstructed, expected)

  def test_fold_with_dilation(self):
    """Verifies k_eff (Effective Kernel) calculation.
    A 3x3 kernel with dilation 2 acts like a 5x5 kernel physically."""
    image_size, kernel_size, stride, dilation = 7, 3, 1, 2
    x = random_ops.random_normal([1, image_size, image_size, 2],seed=42)
    
    patches = self._extract_patches(
        x, kernel=kernel_size,
          stride=stride, 
          padding='VALID', 
          dilation=dilation)
    
    reconstructed = array_ops.fold(
        patches,
        output_size=(image_size, image_size),
        sizes=kernel_size,
        strides=stride,
        padding='VALID',
        rates=dilation
    )

    ones = array_ops.ones_like(x)
    ones_patches = self._extract_patches(
        ones, kernel=kernel_size, 
        stride=stride, 
        padding='VALID', 
        dilation=dilation)
    overlap_counts = array_ops.fold(
        ones_patches,
        output_size=(image_size, image_size),
        sizes=kernel_size,
        strides=stride,
        padding='VALID',
        rates=dilation
    )
    
    expected = x * overlap_counts
    self.assertAllClose(reconstructed, expected)
    
  def test_fold_mixed_symmetric_asymmetric_same_padding(self):
    """Height padding is symmetric, width padding is asymmetric.

    Input: 5x4
    Kernel: 3x3
    Stride: 2x2

    Height:
      out_h = ceil(5/2) = 3
      pad_h = (3 - 1) * 2 + 3 - 5 = 2
      => top=1, bottom=1

    Width:
      out_w = ceil(4/2) = 2
      pad_w = (2 - 1) * 2 + 3 - 4 = 1
      => left=0, right=1
    """
    height, width = 5, 4
    kernel_size, stride = 3, 2

    x = random_ops.random_normal([1, height, width, 1],seed=42)

    patches = self._extract_patches(
        x,
        kernel=kernel_size,
        stride=stride,
        padding="SAME"
    )

    reconstructed = array_ops.fold(
        patches,
        output_size=(height, width),
        sizes=kernel_size,
        strides=stride,
        padding="SAME"
    )

    ones = array_ops.ones_like(x)
    ones_patches = self._extract_patches(
        ones,
        kernel=kernel_size,
        stride=stride,
        padding="SAME"
    )

    overlap_counts = array_ops.fold(
        ones_patches,
        output_size=(height, width),
        sizes=kernel_size,
        strides=stride,
        padding="SAME"
    )

    expected = x * overlap_counts
    self.assertAllClose(reconstructed, expected)
def test_fold_same_padding_with_dilation(self):
  """TEST: SAME padding with dilation > 1.
  kernel=3, dilation=2 -> effective kernel size = 5.
  """
  image_size, kernel_size, stride, dilation = 6, 3, 2, 2

  x = random_ops.random_normal([1, image_size, image_size, 1],seed=42)

  patches = self._extract_patches(
      x,
      kernel=kernel_size,
      stride=stride,
      padding="SAME",
      dilation=dilation)

  reconstructed = array_ops.fold(
      patches,
      output_size=(image_size, image_size),
      sizes=kernel_size,
      strides=stride,
      padding="SAME",
      rates=dilation)

  ones = array_ops.ones_like(x)
  ones_patches = self._extract_patches(
      ones,
      kernel=kernel_size,
      stride=stride,
      padding="SAME",
      dilation=dilation)

  overlap_counts = array_ops.fold(
      ones_patches,
      output_size=(image_size, image_size),
      sizes=kernel_size,
      strides=stride,
      padding="SAME",
      rates=dilation)

  expected = x * overlap_counts
  self.assertAllClose(reconstructed, expected)
  
def test_fold_non_square_parameters(self):
  """TEST: Non-square kernel, stride and dilation."""
  height, width = 7, 8
  kernel_size = (3, 5)
  stride = (2, 3)
  dilation = (1, 2)

  x = random_ops.random_normal([1, height, width, 2],seed=42)

  patches = self._extract_patches(
      x,
      kernel=kernel_size,
      stride=stride,
      padding="SAME",
      dilation=dilation)

  reconstructed = array_ops.fold(
      patches,
      output_size=(height, width),
      sizes=kernel_size,
      strides=stride,
      padding="SAME",
      rates=dilation)

  ones = array_ops.ones_like(x)
  ones_patches = self._extract_patches(
      ones,
      kernel=kernel_size,
      stride=stride,
      padding="SAME",
      dilation=dilation)

  overlap_counts = array_ops.fold(
      ones_patches,
      output_size=(height, width),
      sizes=kernel_size,
      strides=stride,
      padding="SAME",
      rates=dilation)

  expected = x * overlap_counts
  self.assertAllClose(reconstructed, expected)

@test_util.run_all_in_graph_and_eager_modes
class TestFoldDeterminism(test.TestCase):

  def setUp(self):
    super().setUp()
    random_seed.set_seed(42)

  def _extract_patches(self, x, kernel, stride,
                       padding="VALID", dilation=1):
    return array_ops.extract_image_patches_v2(
        images=x,
        sizes=[1, kernel, kernel, 1],
        strides=[1, stride, stride, 1],
        rates=[1, dilation, dilation, 1],
        padding=padding,
    )
  @test_util.run_gpu_only
  def test_fold_gpu_deterministic(self):
    """To check if GPU output is deterministic if op determinism is turned off"""
    config.disable_op_determinism()
    x = random_ops.random_normal([4, 128, 128, 16],dtype=dtypes.float32)
    patches = self._extract_patches(
        x,
        kernel=15,
        stride=1,
        padding="VALID")
    outputs = []
    for i in range(20):
      outputs.append(
        array_ops.fold(
          patches,
          output_size=(128, 128),
          sizes=15,
          strides=1,
          padding="VALID",
          rates=1))
    reference = outputs[0]
    for output in outputs[1:]:
      self.assertAllClose(reference,output)

  def test_fold_cpu_deterministic(self):
    """To check if CPU output is deterministic if op determinism is turned off"""
    config.disable_op_determinism()
    with ops.device("/CPU:0"):
      x = random_ops.random_normal([2, 32, 32, 4],dtype=dtypes.float32)

      patches = self._extract_patches(
          x,
          kernel=5,
          stride=1,
          padding="VALID")

      reference = array_ops.fold(
          patches,
          output_size=(32, 32),
          sizes=5,
          strides=1,
          padding="VALID")

      for i in range(10):
        result = array_ops.fold(
            patches,
            output_size=(32, 32),
            sizes=5,
            strides=1,
            padding="VALID")
        
        self.assertAllClose(reference, result)

if __name__ == "__main__":
  test.main()
