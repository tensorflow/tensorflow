# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for Split Op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

_TEST_DTYPES = (dtypes.float32, dtypes.float64, dtypes.complex64,
                dtypes.complex128)


class SplitOpTest(test.TestCase):

  def _makeData(self, shape, dtype):
    data = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 1j * data
    return data

  @test_util.run_deprecated_v1
  def testShapeInference(self):
    model_input = array_ops.placeholder(dtypes.float32, shape=(1, 10))

    # check that we fail during static shape inference if sizes are known
    with self.assertRaises(ValueError):
      # pylint: disable=expression-not-assigned
      array_ops.split(model_input, [4], axis=1)[0]
      # pylint: enable=expression-not-assigned

    model_input = array_ops.placeholder(dtypes.float32)
    inp = np.zeros((1, 10))
    # check that we still fail at runtime if the shapes were unknown
    with self.cached_session() as sess:
      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(array_ops.split(model_input, [4]), {model_input: inp})

    # scalar Tensors are not permitted as num_splits
    for axis in [0, -2]:
      with self.cached_session() as sess:
        with self.assertRaises(ValueError):
          # pylint: disable=expression-not-assigned
          sess.run(
              array_ops.split(
                  array_ops.ones([4, 4]),
                  num_or_size_splits=constant_op.constant(2),
                  axis=axis))
          # pylint: enable=expression-not-assigned

    # test that none split dimensions remain, even if we don't know how
    # the split_dim will be split, but we do know the axis
    result = array_ops.split(
        array_ops.ones([5, 2]), array_ops.constant([2, 1, 2]) * 1, axis=0)

    self.assertEqual(result[0].shape[1], 2)
    self.assertEqual(result[1].shape[1], 2)
    self.assertEqual(result[2].shape[1], 2)

    model_input2 = array_ops.placeholder(dtypes.float32, shape=[None, 2])
    result = array_ops.split(model_input2, [2, 2], axis=0)[0]

    with self.cached_session() as sess:
      sess.run(result, feed_dict={model_input2: np.ones([4, 2])})

  @test_util.run_deprecated_v1
  def testFailWithoutExplicitNum(self):
    size_splits = array_ops.placeholder(dtype=dtypes.int32, shape=[None])

    value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with self.session() as sess:
      with self.assertRaises(ValueError) as context:
        sess.run(array_ops.split(value, size_splits), {size_splits: [2, 2, 6]})
      self.assertTrue("Cannot infer num from shape" in str(context.exception))

  @test_util.run_in_graph_and_eager_modes
  def testExplicitNum(self):
    size_splits = array_ops.constant([2, 2, 6], dtype=dtypes.int32)
    value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Eager and Graph modes raise different exceptions
    with self.assertRaises((errors_impl.InvalidArgumentError, ValueError)):
      array_ops.split(value, size_splits, num=4)

    r = self.evaluate(array_ops.split(value, size_splits, num=3))
    self.assertAllEqual(r[0], value[0:2])
    self.assertAllEqual(r[1], value[2:4])
    self.assertAllEqual(r[2], value[4:])

  @test_util.run_in_graph_and_eager_modes
  def testListOfScalarTensors(self):
    a = math_ops.cast(5, dtypes.int32)
    b = math_ops.cast(6, dtypes.int32)

    value = np.random.rand(11, 11)

    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(value, [a, b]))

    self.assertAllEqual(result[0], value[0:5, :])
    self.assertAllEqual(result[1], value[5:, :])

  def _RunAndVerifyVariable(self, dtype, large_num_splits=False):
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    split_dim = np.random.randint(-5, 5)
    if large_num_splits:
      num_split = np.random.randint(16, 25)
    else:
      num_split = np.random.randint(2, 8)
    size_splits = np.random.randint(2, 8, num_split, dtype=np.int32)
    shape[split_dim] = np.sum(size_splits)
    inp = self._makeData(shape, dtype)
    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(inp, size_splits, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + size_splits[i])
      offset += size_splits[i]
      self.assertAllEqual(result[i], inp[slices])

  def _testSpecialCasesVariable(self):
    inp = np.random.rand(4, 4).astype("f")

    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(inp, [4], 0))
      self.assertAllEqual(result[0], inp)

      result = self.evaluate(array_ops.split(inp, [-1, 3], 0))
      self.assertAllEqual(result[0], inp[0:1, :])
      self.assertAllEqual(result[1], inp[1:4, :])

  def _testHugeNumberOfTensorsVariable(self, dtype):
    num_split = 1000
    size_splits = np.random.randint(1, 3, num_split, dtype=np.int32)
    shape = [3, np.sum(size_splits)]
    split_dim = 1
    inp = self._makeData(shape, dtype)
    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(inp, size_splits, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + size_splits[i])
      offset += size_splits[i]
      self.assertAllEqual(result[i], inp[slices])

  @test_util.run_in_graph_and_eager_modes
  def testSpecialCasesVariable(self):
    self._testSpecialCasesVariable()
    for dtype in _TEST_DTYPES:
      self._testHugeNumberOfTensorsVariable(dtype)

  @test_util.run_in_graph_and_eager_modes
  def testDegenerateVariable(self):
    inp = np.random.rand(4, 4).astype("f")
    with test_util.device(use_gpu=True):
      result = self.evaluate(array_ops.split(inp, [-1, 4], 0))
      self.assertAllEqual(result[0], inp[0:0, :])
      self.assertAllEqual(result[1], inp[0:4, :])

      result = self.evaluate(array_ops.split(inp, [4, -1], 0))
      self.assertAllEqual(result[0], inp[0:4, :])
      self.assertAllEqual(result[1], inp[4:4, :])

      result = self.evaluate(array_ops.split(inp, [-1, 4], 1))
      self.assertAllEqual(result[0], inp[:, 0:0])
      self.assertAllEqual(result[1], inp[:, 0:4])

      result = self.evaluate(array_ops.split(inp, [4, -1], 1))
      self.assertAllEqual(result[0], inp[:, 0:4])
      self.assertAllEqual(result[1], inp[:, 4:4])

  def _testGradientsSimpleVariable(self, dtype):
    inp = self._makeData((4, 4), dtype)
    with test_util.device(use_gpu=True):
      inp_tensor = ops.convert_to_tensor(inp)
      s = array_ops.split(inp_tensor, [1, 3], 1)
      inp_grads = [
          self._makeData((4, 1), dtype), self._makeData((4, 3), dtype)
      ]
      grad_tensors = [constant_op.constant(x) for x in inp_grads]
      grad = gradients_impl.gradients(s, [inp_tensor], grad_tensors)[-1]
      result = self.evaluate(grad)

    self.assertAllEqual(result[:, 0:1], inp_grads[0])
    self.assertAllEqual(result[:, 1:4], inp_grads[1])

  @test_util.run_deprecated_v1
  def testOutputShape(self):
    for axis in [1, -1]:
      with self.cached_session():
        tensor = array_ops.placeholder(dtypes.float32, shape=[None, 12])
        size_splits = [3, 7, 2]
        outputs = array_ops.split(tensor, size_splits, axis)
        for i, output in enumerate(outputs):
          self.assertEqual(output.get_shape().as_list(), [None, size_splits[i]])

  def _compare(self, x, dim, num):
    np_ans = np.split(x, num, dim)
    with test_util.device(use_gpu=True):
      tf_ans = array_ops.split(value=x, num_or_size_splits=num, axis=dim)
      out = self.evaluate(tf_ans)
    self.assertEqual(num, len(np_ans))
    self.assertEqual(num, len(np_ans))
    self.assertEqual(num, len(out))
    for i in range(num):
      self.assertAllEqual(np_ans[i], out[i])
      self.assertShapeEqual(np_ans[i], tf_ans[i])

  @test_util.run_in_graph_and_eager_modes
  def testSplitRows(self):
    for dtype in _TEST_DTYPES:
      inp = self._makeData((4, 4), dtype)
      self._compare(inp, 0, 4)

  @test_util.run_in_graph_and_eager_modes
  def testSplitCols(self):
    for dtype in _TEST_DTYPES:
      inp = self._makeData((4, 4), dtype)
      self._compare(inp, 1, 4)

  def _testEmpty(self, x, dim, num, expected_shape):
    with test_util.device(use_gpu=True):
      tf_ans = array_ops.split(value=x, num_or_size_splits=num, axis=dim)
      out = self.evaluate(tf_ans)
    self.assertEqual(x.size, 0)
    self.assertEqual(len(out), num)
    for i in range(num):
      self.assertEqual(out[i].shape, expected_shape)
      self.assertEqual(expected_shape, tf_ans[i].get_shape())

  @test_util.run_in_graph_and_eager_modes
  def testEmpty(self):
    # Note: np.split returns a rank-0 empty ndarray
    # if the input ndarray is empty.
    for dtype in _TEST_DTYPES:
      inp = self._makeData((8, 0, 21), dtype)
      self._testEmpty(inp, 0, 2, (4, 0, 21))
      self._testEmpty(inp, 0, 4, (2, 0, 21))
      self._testEmpty(inp, 1, 4, (8, 0, 21))
      self._testEmpty(inp, 2, 3, (8, 0, 7))
      self._testEmpty(inp, 2, 7, (8, 0, 3))

  @test_util.run_in_graph_and_eager_modes
  def testIdentity(self):
    for dtype in _TEST_DTYPES:
      inp = self._makeData((2, 2, 2), dtype)
      self._compare(inp, 0, 1)
      self._compare(inp, 1, 1)
      self._compare(inp, 2, 1)

  @test_util.run_in_graph_and_eager_modes
  def testSplitDim0(self):
    for dtype in _TEST_DTYPES:
      self._compare(self._makeData((6, 10, 18), dtype), 0, 3)
      self._compare(self._makeData((6, 7, 18), dtype), 0, 3)
      self._compare(self._makeData((6, 7, 9), dtype), 0, 3)

  def _RunAndVerify(self, dtype, large_num_splits=False):
    # Random dims of rank 5
    shape = np.random.randint(0, 5, size=5)
    split_dim = np.random.randint(-5, 5)
    if large_num_splits:
      num_split = np.random.randint(9, 15)
    else:
      num_split = np.random.randint(2, 8)
    shape[split_dim] = np.random.randint(2, 5) * num_split
    inp = self._makeData(shape, dtype)
    with test_util.device(use_gpu=True):
      result = self.evaluate(
          array_ops.split(
              value=inp, num_or_size_splits=num_split, axis=split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    length = shape[split_dim] // num_split
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + length)
      offset += length
      self.assertAllEqual(result[i], inp[slices])

  @test_util.run_in_graph_and_eager_modes
  def testRandom(self):
    for dtype in _TEST_DTYPES:
      for _ in range(5):
        self._RunAndVerify(dtype)
        self._RunAndVerify(dtype, large_num_splits=True)
        self._RunAndVerifyVariable(dtype)
        self._RunAndVerifyVariable(dtype, large_num_splits=True)

  def _testGradientsSimple(self, dtype):
    inp = self._makeData((4, 4), dtype)
    with self.cached_session():
      inp_tensor = ops.convert_to_tensor(inp)
      s = array_ops.split(value=inp_tensor, num_or_size_splits=4, axis=1)
      inp_grads = [self._makeData((4, 1), dtype)for _ in range(4)]
      grad_tensors = [constant_op.constant(x) for x in inp_grads]
      grad = gradients_impl.gradients(s, [inp_tensor], grad_tensors)[0]
      result = self.evaluate(grad)
    for i in range(4):
      self.assertAllEqual(result[:, i:i + 1], inp_grads[i])

  @test_util.run_deprecated_v1
  def testGradientsAll(self):
    for dtype in _TEST_DTYPES:
      self._testGradientsSimple(dtype)
      self._testGradientsSimpleVariable(dtype)

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # split_dim greater than rank of input.
    with self.assertRaises(ValueError):
      array_ops.split(value=[[0, 1], [2, 3]], num_or_size_splits=4, axis=2)

    # split dim less than -(rank of input)
    with self.assertRaises(ValueError):
      array_ops.split(value=[[0, 1], [2, 3]], num_or_size_splits=4, axis=-3)

    # num_split does not evenly divide the size in split_dim.
    with self.assertRaisesRegex(ValueError, "should evenly divide"):
      array_ops.split(value=[0, 1, 2, 3], num_or_size_splits=3, axis=0)

    # Unknown split_dim.
    splits = array_ops.split(
        value=[[0, 1, 2, 3]],
        num_or_size_splits=4,
        axis=array_ops.placeholder(dtypes.int32))
    for s in splits:
      self.assertEqual([None, None], s.get_shape().as_list())

    # Unknown split_dim and input shape.
    splits = array_ops.split(
        value=array_ops.placeholder(dtypes.float32),
        num_or_size_splits=4,
        axis=array_ops.placeholder(dtypes.int32))
    for s in splits:
      self.assertEqual(None, s.get_shape().ndims)

  @test_util.run_deprecated_v1
  def testVariableShapeFunction(self):
    # size_splits too big
    with self.assertRaises(ValueError):
      array_ops.split([0, 1], [3, -1], axis=0)

    # Correct inference of variable dimension
    s0, s1 = array_ops.split([0, 1, 2], [2, -1], axis=0)
    assert s0.shape.as_list() == [2]
    assert s1.shape.as_list() == [1]

  @test_util.run_deprecated_v1
  def testNonexistentDimTensor(self):
    x = array_ops.placeholder(dtypes.int32)
    values = np.zeros([5, 30])
    splits = array_ops.placeholder(dtypes.int32)
    with self.assertRaisesRegex(ValueError, "Cannot infer"):
      y = array_ops.split(values, splits, axis=x)

    splits = array_ops.placeholder(dtypes.int32, [3])
    y = array_ops.split(values, splits, axis=x)
    with self.session() as sess:
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "must have exactly one element"):
        sess.run(y, {x: np.array([], dtype=np.int32), splits: [4, 11, 15]})


if __name__ == "__main__":
  test.main()
