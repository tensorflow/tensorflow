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


if __name__ == "__main__":
  test.main()
