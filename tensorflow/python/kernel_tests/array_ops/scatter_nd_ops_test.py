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
"""Tests for tensorflow.ops.tf.scatter_nd."""

import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

GRADIENT_TESTS_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64)


def _AsType(v, vtype):
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _FlatInnerDims(tensor, ndims=2):
  shape = list(tensor.shape)
  return tensor.reshape(
      [functools.reduce(lambda x, y: x * y, shape[:-ndims + 1], 1)] +
      shape[-ndims + 1:])


def _FlatOuterDims(tensor, ndims=2):
  shape = list(tensor.shape)
  return tensor.reshape(
      shape[:ndims - 1] +
      [functools.reduce(lambda x, y: x * y, shape[ndims - 1:], 1)])


def _NumpyScatterNd(ref, indices, updates, op):
  ixdim = indices.shape[-1]
  num_updates = indices.size // ixdim
  total_nd = len(ref.shape)
  slice_size = 1
  for i in range(ixdim, total_nd):
    slice_size *= ref.shape[i]
  flat_indices = _FlatInnerDims(indices)
  flat_updates = updates.reshape((num_updates, slice_size))
  output_flat = _FlatOuterDims(ref, ixdim + 1)
  for ix_updates, ix_output in enumerate(flat_indices):
    ix_output = tuple(ix_output)
    output_flat[ix_output] = op(output_flat[ix_output],
                                flat_updates[ix_updates])
  return output_flat.reshape(ref.shape)


def _NumpyUpdate(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: u)


def _NumpyAdd(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p + u)


def _NumpySub(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p - u)


def _NumpyMul(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p * u)


def _NumpyDiv(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p / u)


def _NumpyMin(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, np.minimum)


def _NumpyMax(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, np.maximum)


@test_util.with_eager_op_as_function
class StatefulScatterNdTest(test.TestCase):

  def _VariableRankTest(self,
                        np_scatter,
                        tf_scatter,
                        vtype,
                        itype,
                        repeat_indices=False):
    np.random.seed(8)
    ref_shapes = [(3, 6), (3, 6), (3, 6, 9), (3, 6, 9), (3, 6, 9), (3, 6, 9)]
    indices_shapes = [(2,), (2, 2), (2,), (2, 2), (2, 3), (2, 3, 3)]
    with test_util.device(use_gpu=True):
      for ref_shape, indices_shape in zip(ref_shapes, indices_shapes):
        num_updates = indices_shape[0]
        ixdim = indices_shape[-1]

        indexable_area_shape = ()
        for i in range(ixdim):
          indexable_area_shape += (ref_shape[i],)
        all_indices = [
            list(coord) for coord, _ in np.ndenumerate(
                np.empty(indexable_area_shape, vtype))
        ]
        np.random.shuffle(all_indices)
        indices = np.array(all_indices[:num_updates])

        if num_updates > 1 and repeat_indices:
          indices = indices[:num_updates // 2]
          for _ in range(num_updates - num_updates // 2):
            indices = np.append(
                indices, [indices[np.random.randint(num_updates // 2)]], axis=0)
          np.random.shuffle(indices)
        indices = _AsType(indices[:num_updates], itype)

        updates_shape = (num_updates,)
        for i in range(ixdim, len(ref_shape)):
          updates_shape += (ref_shape[i],)
        updates = _AsType(np.random.randn(*(updates_shape)), vtype)
        ref = _AsType(np.random.randn(*(ref_shape)), vtype)

        # Scatter via numpy
        new = ref.copy()
        np_scatter(new, indices, updates)
        # Scatter via tensorflow
        ref_var = variables.VariableV1(ref)
        self.evaluate(ref_var.initializer)
        self.evaluate(tf_scatter(ref_var, indices, updates))

        # Compare
        self.assertAllClose(new, self.evaluate(ref_var))

  def _VariableRankTests(self, np_scatter, tf_scatter):
    for vtype in (np.int32, np.float16, np.float32, np.float64, np.complex64,
                  np.complex128):
      for itype in (np.int32, np.int64):
        self._VariableRankTest(np_scatter, tf_scatter, vtype, itype)

  def testSimple(self):
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    for dtype in (dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64,
                  dtypes.complex64, dtypes.complex128):
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtype)
      ref = variables.Variable([0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype)
      expected = np.array([0, 11, 0, 10, 9, 0, 0, 12])
      scatter = state_ops.scatter_nd_update(ref, indices, updates)
      init = variables.global_variables_initializer()

      with test_util.use_gpu():
        self.evaluate(init)
        result = self.evaluate(scatter)
        self.assertAllClose(result, expected)

  @test_util.run_in_graph_and_eager_modes
  def testString(self):
    ref = variables.Variable(["qq", "ww", "ee", "rr", "", "", "", ""])
    indices = constant_op.constant([[4], [3], [1], [7]])
    updates = constant_op.constant(["aa", "dd", "cc", "bb"])
    update = state_ops.scatter_nd_update(ref, indices, updates)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(update),
        [b"qq", b"cc", b"ee", b"dd", b"aa", b"", b"", b"bb"])

  def testSimpleResource(self):
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    for dtype in (dtypes.int32, dtypes.float32):
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtype)
      ref = resource_variable_ops.ResourceVariable([0, 0, 0, 0, 0, 0, 0, 0],
                                                   dtype=dtype)
      expected = np.array([0, 11, 0, 10, 9, 0, 0, 12])
      scatter = state_ops.scatter_nd_update(ref, indices, updates)

      with test_util.device(use_gpu=True):
        self.evaluate(ref.initializer)
        self.evaluate(scatter)
        self.assertAllClose(ref, expected)

  def testSimple2(self):
    indices = constant_op.constant([[1, 0], [1, 1]], dtype=dtypes.int32)
    updates = constant_op.constant([11., 12.], dtype=dtypes.float32)
    ref = variables.Variable([[0., 0.], [0., 0.], [0., 0.]],
                             dtype=dtypes.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]])
    scatter = state_ops.scatter_nd_update(ref, indices, updates)
    init = variables.global_variables_initializer()

    with self.session():
      self.evaluate(init)
      result = self.evaluate(scatter)
      self.assertAllClose(result, expected)

  def testSimple3(self):
    indices = constant_op.constant([[1]], dtype=dtypes.int32)
    updates = constant_op.constant([[11., 12.]], dtype=dtypes.float32)
    ref = variables.Variable([[0., 0.], [0., 0.], [0., 0.]],
                             dtype=dtypes.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]])
    scatter = state_ops.scatter_nd_update(ref, indices, updates)
    init = variables.global_variables_initializer()

    with self.session():
      self.evaluate(init)
      result = self.evaluate(scatter)
      self.assertAllClose(result, expected)

  def testVariableRankUpdate(self):
    self._VariableRankTests(_NumpyUpdate, state_ops.scatter_nd_update)

  def testVariableRankAdd(self):
    self._VariableRankTests(_NumpyAdd, state_ops.scatter_nd_add)

  def testVariableRankSub(self):
    self._VariableRankTests(_NumpySub, state_ops.scatter_nd_sub)

  # TODO(ebrevdo): Re-enable when we need ScatterNdMul.
  # def testVariableRankMul(self):
  #   self._VariableRankTests(_NumpyMul, state_ops.scatter_nd_mul)

  # TODO(ebrevdo): Re-enable when we need ScatterNdDiv.
  # def testVariableRankDiv(self):
  #   self._VariableRankTests(_NumpyDiv, state_ops.scatter_nd_div)

  def _ScatterRepeatIndicesTest(self, np_scatter, tf_scatter):
    for vtype in (np.int32, np.float16, np.float32, np.float64):
      for itype in (np.int32, np.int64):
        self._VariableRankTest(
            np_scatter, tf_scatter, vtype, itype, repeat_indices=True)

  def testScatterRepeatIndices(self):
    """This tests scatter_add using indices that repeat."""
    self._ScatterRepeatIndicesTest(_NumpyAdd, state_ops.scatter_nd_add)
    self._ScatterRepeatIndicesTest(_NumpySub, state_ops.scatter_nd_sub)
    # TODO(ebrevdo): Re-enable when we need ScatterNdMul and ScatterNdDiv.
    # self._ScatterRepeatIndicesTest(_NumpyMul, state_ops.scatter_nd_mul)
    # self._ScatterRepeatIndicesTest(_NumpyDiv, state_ops.scatter_nd_div)

  # TODO(simister): Re-enable once binary size increase due to
  # extra templating is back under control and this op is re-enabled
  # def testBooleanScatterUpdate(self):
  #   with self.session(use_gpu=False) as session:
  #     var = tf.Variable([True, False])
  #     update0 = tf.compat.v1.scatter_nd_update(var, [[1]], [True])
  #     update1 = tf.compat.v1.scatter_nd_update(
  #         var, tf.constant(
  #             [[0]], dtype=tf.int64), [False])
  #     self.evaluate(var.initializer)
  #     session.run([update0, update1])
  #     self.assertAllEqual([False, True], self.evaluate(var))

  @test_util.disable_xla("b/205330448")
  def testScatterOutOfRangeCpu(self):
    # TODO(simister): Re-enable once binary size increase due to
    # scatter_nd ops is under control.
    #  tf.scatter_nd_mul, tf.scatter_nd_div,
    for op in (state_ops.scatter_nd_add, state_ops.scatter_nd_sub,
               state_ops.scatter_nd_update):
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      with test_util.device(use_gpu=False):
        ref = variables.VariableV1(params)
        self.evaluate(ref.initializer)

        # Indices all in range, no problem.
        indices = np.array([[2], [0], [5]])
        self.evaluate(op(ref, indices, updates))

        # Test some out of range errors.
        indices = np.array([[-1], [0], [5]])
        with self.assertRaisesOpError(
            r"indices\[0\] = \[-1\] does not index into shape \[6\]"):
          self.evaluate(op(ref, indices, updates))

        indices = np.array([[2], [0], [6]])
        with self.assertRaisesOpError(
            r"indices\[2\] = \[6\] does not index into shape \[6\]"):
          self.evaluate(op(ref, indices, updates))

  def testRank3ValidShape(self):
    indices = array_ops.zeros([2, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    self.assertAllEqual(
        state_ops.scatter_nd_update(ref, indices,
                                    updates).get_shape().as_list(), shape)

  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testResVarInvalidOutputShape(self):
    res = variables.Variable(
        initial_value=lambda: array_ops.zeros(shape=[], dtype=dtypes.float32),
        dtype=dtypes.float32)
    with self.cached_session():
      self.evaluate(res.initializer)
      with self.assertRaisesOpError("Output must be at least 1-D"):
        state_ops.scatter_nd_update(res, [[0]], [0.22]).eval()

  def testExtraIndicesDimensions(self):
    indices = array_ops.zeros([1, 1, 2], dtypes.int32)
    updates = array_ops.zeros([1, 1], dtypes.int32)
    shape = np.array([2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    scatter_update = state_ops.scatter_nd_update(ref, indices, updates)
    self.assertAllEqual(scatter_update.get_shape().as_list(), shape)

    expected_result = np.zeros([2, 2], dtype=np.int32)
    with self.cached_session():
      self.evaluate(ref.initializer)
      self.assertAllEqual(expected_result, self.evaluate(scatter_update))

  def testRank3InvalidShape1(self):
    indices = array_ops.zeros([3, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        r"Dimensions \[\d,\d\) of indices\[shape="):
      state_ops.scatter_nd_update(ref, indices, updates)

  def testRank3InvalidShape2(self):
    indices = array_ops.zeros([2, 2, 1], dtypes.int32)
    updates = array_ops.zeros([2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        r"Dimensions \[\d,\d\) of input\[shape="):
      state_ops.scatter_nd_update(ref, indices, updates)

  def testConcurrentUpdates(self):
    num_updates = 10000
    update_values = np.random.rand(num_updates)
    ref = variables.Variable(np.zeros([2, 2]), dtype=dtypes.float64)
    indices = constant_op.constant([[0, 1]] * num_updates, dtype=dtypes.int32)
    updates = constant_op.constant(update_values, dtype=dtypes.float64)

    expected_result = np.zeros([2, 2], dtype=np.float64)
    expected_result[0, 1] = np.sum(update_values)

    scatter = state_ops.scatter_nd_add(ref, indices, updates)
    init = variables.global_variables_initializer()

    self.evaluate(init)
    result = self.evaluate(scatter)
    assert np.allclose(result, expected_result)

  @test_util.run_in_graph_and_eager_modes
  def testMin(self):
    variable = variables.Variable(array_ops.ones([8], dtype=dtypes.int32))
    resource_variable = resource_variable_ops.ResourceVariable(
        array_ops.ones([8], dtype=dtypes.int32))
    indices = constant_op.constant([4, 3, 1, 7])
    updates = constant_op.constant([0, 2, -1, 2], dtype=dtypes.int32)

    for ref in (variable, resource_variable):
      min_result = state_ops.scatter_min(ref, indices, updates)
      self.evaluate(ref.initializer)

      expected_result = constant_op.constant([1, -1, 1, 1, 0, 1, 1, 1])
      self.assertAllEqual(self.evaluate(min_result), expected_result)
      self.assertAllEqual(self.evaluate(ref), expected_result)

  @test_util.run_in_graph_and_eager_modes
  def testMax(self):
    variable = variables.Variable(array_ops.ones([8], dtype=dtypes.int32))
    resource_variable = resource_variable_ops.ResourceVariable(
        array_ops.ones([8], dtype=dtypes.int32))
    indices = constant_op.constant([4, 3, 1, 7])
    updates = constant_op.constant([0, 2, -1, 2], dtype=dtypes.int32)

    for ref in (variable, resource_variable):
      max_result = state_ops.scatter_max(ref, indices, updates)
      self.evaluate(ref.initializer)

      expected_result = constant_op.constant([1, 1, 1, 2, 1, 1, 1, 2])
      self.assertAllEqual(self.evaluate(max_result), expected_result)
      self.assertAllEqual(self.evaluate(ref), expected_result)

  @test_util.run_in_graph_and_eager_modes
  def testAdd(self):
    variable = variables.Variable(array_ops.ones([8], dtype=dtypes.int32))
    resource_variable = resource_variable_ops.ResourceVariable(
        array_ops.ones([8], dtype=dtypes.int32))
    indices = constant_op.constant([4, 3, 1, 7])
    updates = constant_op.constant([0, 2, -1, 3], dtype=dtypes.int32)

    for ref in (variable, resource_variable):
      add_result = state_ops.scatter_add(ref, indices, updates)
      self.evaluate(ref.initializer)

      expected_result = constant_op.constant([1, 0, 1, 3, 1, 1, 1, 4])
      self.assertAllEqual(self.evaluate(add_result), expected_result)
      self.assertAllEqual(self.evaluate(ref), expected_result)

  @test_util.run_in_graph_and_eager_modes
  def testSub(self):
    variable = variables.Variable(array_ops.ones([8], dtype=dtypes.int32))
    resource_variable = resource_variable_ops.ResourceVariable(
        array_ops.ones([8], dtype=dtypes.int32))
    indices = constant_op.constant([4, 3, 1, 7])
    updates = constant_op.constant([0, 2, -1, 2], dtype=dtypes.int32)

    for ref in (variable, resource_variable):
      sub_result = state_ops.scatter_sub(ref, indices, updates)
      self.evaluate(ref.initializer)

      expected_result = constant_op.constant([1, 2, 1, -1, 1, 1, 1, -1])
      self.assertAllEqual(self.evaluate(sub_result), expected_result)
      self.assertAllEqual(self.evaluate(ref), expected_result)

  # TODO(fpmc): Re-enable this test when gpu_pip test actually runs on a GPU.
  def _disabledTestScatterOutOfRangeGpu(self):
    if not test.IsBuiltWithCuda():
      return
    # TODO(simister): Re-enable once binary size increase due to
    # scatter_nd ops is under control.
    # tf.scatter_nd_mul, tf.scatter_nd_div,
    for op in (state_ops.scatter_nd_add, state_ops.scatter_nd_sub,
               state_ops.scatter_nd_update):
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      # With GPU, the code ignores indices that are out of range.
      # We don't test the implementation; just test there's no failures.
      with self.cached_session(force_gpu=True):
        ref = variables.Variable(params)
        self.evaluate(ref.initializer)

        # Indices all in range, no problem.
        indices = np.array([2, 0, 5])
        op(ref, indices, updates).eval()

        # Indices out of range should not fail.
        indices = np.array([-1, 0, 5])
        op(ref, indices, updates).eval()
        indices = np.array([2, 0, 6])
        op(ref, indices, updates).eval()


class StatefulScatterNdDeterminismTest(StatefulScatterNdTest):

  def setUp(self):
    super().setUp()
    config.enable_op_determinism()

  def tearDown(self):
    super().tearDown()
    config.disable_op_determinism()

  def testDeterminism(self):
    ref = variables.Variable(array_ops.zeros([1]))
    indices = array_ops.zeros([100000, 1], dtypes.int32)
    values = np.random.randn(100000)
    self.evaluate(variables.global_variables_initializer())
    val = self.evaluate(state_ops.scatter_nd_update(ref, indices, values))
    for _ in range(5):
      ref2 = variables.Variable(array_ops.zeros([1]))
      self.evaluate(variables.global_variables_initializer())
      val2 = self.evaluate(state_ops.scatter_nd_update(ref2, indices, values))
      self.assertAllEqual(val, val2)


@test_util.with_eager_op_as_function
class ScatterNdTest(test.TestCase, parameterized.TestCase):
  non_aliasing_add_test = False

  def scatter_nd(self, indices, updates, shape, input_=None):
    del input_  # input_ is not used in scatter_nd
    return array_ops.scatter_nd(indices, updates, shape)

  @test_util.run_in_graph_and_eager_modes
  def testBool(self):
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    updates = constant_op.constant([False, True, False, True],
                                   dtype=dtypes.bool)
    expected = np.array([False, False, False, True, False, False, False, True])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    result = self.evaluate(scatter)
    self.assertAllEqual(expected, result)

    # Same indice is updated twice by same value.
    indices = constant_op.constant([[4], [3], [3], [7]], dtype=dtypes.int32)
    updates = constant_op.constant([False, True, True, True], dtype=dtypes.bool)
    expected = np.array([False, False, False, True, False, False, False, True])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    result = self.evaluate(scatter)
    self.assertAllEqual(expected, result)

  @test_util.run_in_graph_and_eager_modes
  def testInvalidShape(self):
    # TODO(apassos) figure out how to unify these errors
    with self.assertRaises(errors.InvalidArgumentError if context
                           .executing_eagerly() else ValueError):
      array_ops.scatter_nd(
          indices=[0],  # this should be indices=[[0]]
          updates=[0.0],
          shape=[1])

  def testString(self):
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    updates = constant_op.constant(["four", "three", "one", "seven"],
                                   dtype=dtypes.string)
    expected = np.array(
        [b"", b"one", b"", b"three", b"four", b"", b"", b"seven"])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    with self.cached_session() as sess:
      result = self.evaluate(scatter)
      self.assertAllEqual(expected, result)

    # Same indice is updated twice by same value.
    indices = constant_op.constant([[4], [3], [3], [7]], dtype=dtypes.int32)
    updates = constant_op.constant(["a", "b", "b", "c"], dtype=dtypes.string)
    expected = np.array([b"", b"", b"", b"bb", b"a", b"", b"", b"c"])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    with self.cached_session() as sess:
      result = self.evaluate(scatter)
      self.assertAllEqual(expected, result)

    # Same indice is updated twice by different value.
    indices = constant_op.constant([[4], [3], [3], [7]], dtype=dtypes.int32)
    updates = constant_op.constant(["a", "b", "c", "d"], dtype=dtypes.string)
    expected = [
        np.array([b"", b"", b"", b"bc", b"a", b"", b"", b"d"]),
        np.array([b"", b"", b"", b"cb", b"a", b"", b"", b"d"])
    ]
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    with self.cached_session() as sess:
      result = self.evaluate(scatter)
      self.assertTrue(
          np.array_equal(result, expected[0]) or
          np.array_equal(result, expected[1]))

  def testRank3ValidShape(self):
    indices = array_ops.zeros([2, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    self.assertAllEqual(
        self.scatter_nd(indices, updates, shape).get_shape().as_list(), shape)

  def testExtraIndicesDimensions(self):
    indices = array_ops.zeros([1, 1, 2], dtypes.int32)
    updates = array_ops.zeros([1, 1], dtypes.int32)
    shape = np.array([2, 2])
    scatter = self.scatter_nd(indices, updates, shape)
    self.assertAllEqual(scatter.get_shape().as_list(), shape)
    expected_result = np.zeros([2, 2], dtype=np.int32)
    self.assertAllEqual(expected_result, self.evaluate(scatter))

  def testUndefinedIndicesShape(self):
    # Placeholders are only valid in Graph.
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int32, shape=None)
      updates = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
      shape = constant_op.constant([2, 2, 2], dtypes.int32)
      self.scatter_nd(indices, updates, shape)

  def testUndefinedUpdatesShape(self):
    # Placeholders are only valid in Graph.
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
      updates = array_ops.placeholder(dtypes.int32, shape=None)
      shape = constant_op.constant([2, 2, 2], dtypes.int32)
      self.scatter_nd(indices, updates, shape)

  def testUndefinedOutputShape(self):
    # Placeholders are only valid in Graph.
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
      updates = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
      shape = array_ops.placeholder(dtypes.int32, shape=[None])
      self.scatter_nd(indices, updates, shape)

  def testEmptyOutputShape1(self):
    indices = array_ops.zeros([2, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = constant_op.constant([0, 3, 2], dtypes.int32)

    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        "Indices and updates specified for empty"):
      self.scatter_nd(indices, updates, shape)

  def testEmptyOutputShape2(self):
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int32, shape=None)
      updates = array_ops.placeholder(dtypes.int32, shape=None)
      shape = constant_op.constant([0, 3, 2], dtypes.int32)

      with self.cached_session():
        with self.assertRaisesOpError(
            "Indices and updates specified for empty (input|output)"):
          self.scatter_nd(indices, updates, shape).eval(
              feed_dict={
                  indices: np.zeros([2, 2, 2], dtype=np.int32),
                  updates: np.zeros([2, 2, 2], dtype=np.int32)
              })

  def testEmptyOutputShape3(self):
    indices = array_ops.zeros([0], dtypes.int32)
    updates = array_ops.zeros([0], dtypes.int32)
    shape = constant_op.constant([0], dtypes.int32)
    scatter = self.scatter_nd(indices, updates, shape)

    with self.cached_session():
      self.assertEqual(self.evaluate(scatter).size, 0)

  def testRank3InvalidShape1(self):
    indices = array_ops.zeros([3, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        r"Dimensions \[\d\,\d\) of indices\[shape="):
      self.scatter_nd(indices, updates, shape)

  def testRank3InvalidShape2(self):
    indices = array_ops.zeros([2, 2, 1], dtypes.int32)
    updates = array_ops.zeros([2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        r"Dimensions \[\d\,\d\) of input\[shape="):
      self.scatter_nd(indices, updates, shape)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientsRank2ElementUpdate(self, use_tape):
    for dtype in GRADIENT_TESTS_DTYPES:
      with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
        indices = constant_op.constant([[0, 0], [1, 1]], dtype=dtypes.int32)
        updates = constant_op.constant([1, 4], dtype=dtype)
        tape.watch(updates)
        shape = constant_op.constant([2, 2], dtype=dtypes.int32)
        input_ = array_ops.zeros(shape, dtype=dtype)
        tape.watch(input_)
        outputs = self.scatter_nd(indices, updates, shape, input_)

        grad_vals = constant_op.constant([[1, 2], [3, 4]], dtype=dtype)

        updates_grad, input_grad = tape.gradient([outputs], [updates, input_],
                                                 [grad_vals])
      expected_updates_grad = np.array([1, 4], dtype=dtype.as_numpy_dtype())
      expected_input_grad = np.array([[1, 2], [3, 4]],
                                     dtype=dtype.as_numpy_dtype())
      self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
      if self.non_aliasing_add_test:
        self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientsRank2SliceUpdate(self, use_tape):
    for dtype in GRADIENT_TESTS_DTYPES:
      with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
        indices = constant_op.constant([[1], [0]], dtype=dtypes.int32)
        updates = constant_op.constant([[3, 4], [1, 2]], dtype=dtype)
        tape.watch(updates)
        shape = constant_op.constant([2, 2], dtype=dtypes.int32)
        input_ = array_ops.zeros(shape, dtype=dtype)
        tape.watch(input_)
        outputs = self.scatter_nd(indices, updates, shape, input_)

        grad_vals = constant_op.constant([[3, 4], [1, 2]], dtype=dtype)
        updates_grad, input_grad = tape.gradient([outputs], [updates, input_],
                                                 [grad_vals])
      expected_updates_grad = np.array([[1, 2], [3, 4]],
                                       dtype=dtype.as_numpy_dtype())
      expected_input_grad = np.array([[3, 4], [1, 2]],
                                     dtype=dtype.as_numpy_dtype())
      self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
      if self.non_aliasing_add_test:
        self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientsRank3SliceUpdate(self, use_tape):
    for dtype in GRADIENT_TESTS_DTYPES:
      with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
        indices = constant_op.constant([[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                                       dtype=dtypes.int32)
        updates = constant_op.constant([[[5, 7], [2, 4]], [[1, 3], [6, 8]]],
                                       dtype=dtype)
        tape.watch(updates)
        shape = constant_op.constant([2, 2, 2], dtype=dtypes.int32)
        input_ = array_ops.zeros(shape, dtype=dtype)
        tape.watch(input_)
        outputs = self.scatter_nd(indices, updates, shape, input_)

        grad_vals = constant_op.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                                         dtype=dtype)
        updates_grad, input_grad = tape.gradient([outputs], [updates, input_],
                                                 [grad_vals])
      expected_updates_grad = np.array([[[3, 4], [5, 6]], [[1, 2], [7, 8]]],
                                       dtype=dtype.as_numpy_dtype())
      expected_input_grad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                                     dtype=dtype.as_numpy_dtype())
      self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
      if self.non_aliasing_add_test:
        self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientsRank7SliceUpdate(self, use_tape):
    for dtype in GRADIENT_TESTS_DTYPES:
      with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
        indices = constant_op.constant(
            [[[[[[[0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]]]],
               [[[[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1]]]]]]],
            dtype=dtypes.int32)
        updates = constant_op.constant(
            [[[[[[[5, 6], [2, 4]]]], [[[[1, 3], [6, 8]]]]]]], dtype=dtype)
        tape.watch(updates)
        shape = constant_op.constant([1, 1, 2, 1, 1, 2, 2], dtype=dtypes.int32)
        input_ = array_ops.zeros(shape, dtype=dtype)
        tape.watch(input_)
        outputs = self.scatter_nd(indices, updates, shape, input_)

        grad_vals = constant_op.constant(
            [[[[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]]]], dtype=dtype)
        updates_grad, input_grad = tape.gradient([outputs], [updates, input_],
                                                 [grad_vals])
      expected_updates_grad = np.array(
          [[[[[[[3, 4], [5, 6]]]], [[[[1, 2], [7, 8]]]]]]],
          dtype=dtype.as_numpy_dtype())
      expected_input_grad = np.array(
          [[[[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]]]],
          dtype=dtype.as_numpy_dtype())
      self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
      if self.non_aliasing_add_test:
        self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  def testScatterNdRepeatedIndicesAdd(self):
    indices = array_ops.zeros([100000, 1], dtypes.int32)
    values = np.random.randn(100000)
    shape = [1]
    val = self.evaluate(self.scatter_nd(indices, values, shape))
    self.assertAllClose([np.sum(values)], val)

  def testSmokeScatterNdBatch2DSliceDim2(self):
    indices = array_ops.zeros([3, 5, 2], dtype=dtypes.int32)
    values = array_ops.zeros([3, 5, 7])
    shape = [4, 6, 7]
    self.evaluate(self.scatter_nd(indices, values, shape))

  def testSmokeScatterNdBatch1DSliceDim2(self):
    indices = array_ops.zeros([0, 2], dtype=dtypes.int32)
    values = array_ops.zeros([0, 7])
    shape = [4, 6, 7]
    self.evaluate(self.scatter_nd(indices, values, shape))

  def testSmokeScatterNdBatch1DSliceDim3ShapeRank7(self):
    indices = array_ops.zeros([1, 3], dtype=dtypes.int32)
    values = array_ops.zeros([1, 6, 7, 8, 9])
    shape = [3, 4, 5, 6, 7, 8, 9]
    self.evaluate(self.scatter_nd(indices, values, shape))

  def testSmokeScatterNdBatch2DSliceDim3ShapeRank7(self):
    indices = array_ops.zeros([1, 2, 3], dtype=dtypes.int32)
    values = array_ops.zeros([1, 2, 6, 7, 8, 9])
    shape = [3, 4, 5, 6, 7, 8, 9]
    self.evaluate(self.scatter_nd(indices, values, shape))


class ScatterNdNonAliasingAddTest(ScatterNdTest):
  non_aliasing_add_test = True

  def scatter_nd(self, indices, updates, shape, input_=None):
    input_ = (
        input_ if input_ is not None else array_ops.zeros(
            shape, dtype=updates.dtype))
    return array_ops.scatter_nd_non_aliasing_add(input_, indices, updates)

  def testString(self):
    # Not supported yet.
    pass

  # TODO(testString): Enable this test when the above testString is enabled.
  def testStringWithEagerOpAsFunctionEnabled(self):
    # Not supported yet.
    pass


class ScatterNdDeterminismTest(ScatterNdTest):

  def setUp(self):
    super().setUp()
    config.enable_op_determinism()

  def tearDown(self):
    super().tearDown()
    config.disable_op_determinism()

  def testDeterminism(self):
    indices = array_ops.zeros([100000, 1], dtypes.int32)
    values = np.random.randn(100000)
    shape = [1]
    val = self.evaluate(self.scatter_nd(indices, values, shape))
    for _ in range(5):
      val2 = self.evaluate(self.scatter_nd(indices, values, shape))
      self.assertAllEqual(val, val2)


class ScatterNdNonAliasingAddDeterminismTest(ScatterNdDeterminismTest,
                                             ScatterNdNonAliasingAddTest):
  pass


class ScatterNdTensorTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testUpdateAddSub(self):
    for dtype in (dtypes.int32, dtypes.float32):
      indices = constant_op.constant([[4], [3], [1], [7]])
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtype)
      t = array_ops.ones([8], dtype=dtype)
      assigned = array_ops.tensor_scatter_update(t, indices, updates)
      added = array_ops.tensor_scatter_add(t, indices, updates)
      subbed = array_ops.tensor_scatter_sub(t, indices, updates)

      self.assertAllEqual(assigned,
                          constant_op.constant([1, 11, 1, 10, 9, 1, 1, 12]))
      self.assertAllEqual(added,
                          constant_op.constant([1, 12, 1, 11, 10, 1, 1, 13]))
      self.assertAllEqual(subbed,
                          constant_op.constant([1, -10, 1, -9, -8, 1, 1, -11]))

  def testUpdateAddSubGradients(self):
    with self.cached_session():
      indices = constant_op.constant([[3], [1]])
      updates = constant_op.constant([9, 10], dtype=dtypes.float32)
      x = array_ops.ones([4], dtype=dtypes.float32)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_update(x, indices, updates), [x])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_add(x, indices, updates), [x])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_sub(x, indices, updates), [x])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_update(x, indices, updates),
          [updates])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_add(x, indices, updates),
          [updates])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_sub(x, indices, updates),
          [updates])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)

  @test_util.run_in_graph_and_eager_modes
  def testUpdateMinMax(self):
    for dtype in (dtypes.int32, dtypes.float32):
      indices = constant_op.constant([[4], [3], [1], [7]])
      updates = constant_op.constant([0, 2, -1, 2], dtype=dtype)
      t = array_ops.ones([8], dtype=dtype)
      assigned = array_ops.tensor_scatter_update(t, indices, updates)
      min_result = array_ops.tensor_scatter_min(t, indices, updates)
      max_result = array_ops.tensor_scatter_max(t, indices, updates)

      self.assertAllEqual(assigned,
                          constant_op.constant([1, -1, 1, 2, 0, 1, 1, 2]))
      self.assertAllEqual(min_result,
                          constant_op.constant([1, -1, 1, 1, 0, 1, 1, 1]))
      self.assertAllEqual(max_result,
                          constant_op.constant([1, 1, 1, 2, 1, 1, 1, 2]))

  def testUpdateMinMaxGradients(self):
    with self.cached_session():
      x = array_ops.ones([4], dtype=dtypes.float32)
      indices = constant_op.constant([[1], [2], [3], [3]])
      updates = constant_op.constant([2.0, 0.5, 1.0, 1.0], dtype=dtypes.float32)

      theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_max(x, indices, updates), [x])
      # Numerical gradient doesn't work for degenerate values because the
      # derivative is not continuous. The manually entered gradient divides
      # the gradient among all contributing elements at the discontinuity.
      manual = array_ops.reshape(
          array_ops.matrix_diag([1.0, 0.0, 1.0, 0.3333]), (1, 4, 4))
      self.assertAllClose(theoretical, manual, 5e-4, 5e-4)

      theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_min(x, indices, updates), [x])
      manual = array_ops.reshape(
          array_ops.matrix_diag([1.0, 1.0, 0.0, 0.3333]), (1, 4, 4))
      self.assertAllClose(theoretical, manual, 5e-4, 5e-4)

      theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_max(x, indices, updates),
          [updates])
      manual = constant_op.constant(
          [[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3333, 0.3333]]],
          dtype=dtypes.float32)
      self.assertAllClose(theoretical, manual, 5e-4, 5e-4)

      theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_min(x, indices, updates),
          [updates])
      manual = constant_op.constant(
          [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.3333, 0.3333]]],
          dtype=dtypes.float32)
      self.assertAllClose(theoretical, manual, 5e-4, 5e-4)

  def testTensorScatterUpdateWithForwarding(self):
    for dtype in (dtypes.int32, dtypes.float32):

      @def_function.function
      def _TestFn():
        indices = constant_op.constant([[4], [3], [1], [7]])
        updates = constant_op.constant([9, 10, 11, 12], dtype=dtype)  # pylint: disable=cell-var-from-loop
        t = array_ops.ones([8], dtype=dtype)  # pylint: disable=cell-var-from-loop

        return array_ops.tensor_scatter_update(t, indices, updates)

      self.assertAllEqual(_TestFn(), [1, 11, 1, 10, 9, 1, 1, 12])

  @test_util.run_in_graph_and_eager_modes
  def testTensorScatterUpdateWithStrings(self):
    indices = constant_op.constant([[4], [3], [1], [7]])
    updates = constant_op.constant(["there", "there", "there", "12"],
                                   dtype=dtypes.string)
    tensor = constant_op.constant([
        "hello", "hello", "hello", "hello", "hello", "hello", "hello", "hello"
    ],
                                  dtype=dtypes.string)
    updated = array_ops.tensor_scatter_update(tensor, indices, updates)

    self.assertAllEqual(
        updated,
        constant_op.constant([
            "hello", "there", "hello", "there", "there", "hello", "hello", "12"
        ]))

  @test_util.run_in_graph_and_eager_modes
  def testUpdateRepeatedIndices1D(self):
    if test_util.is_gpu_available():
      self.skipTest("Duplicate indices scatter is non-deterministic on GPU")
    a = array_ops.zeros([10, 1])
    b = array_ops.tensor_scatter_update(a, [[5], [5]], [[4], [8]])
    self.assertAllEqual(
        b,
        constant_op.constant([[0.], [0.], [0.], [0.], [0.], [8.], [0.], [0.],
                              [0.], [0.]]))

  @test_util.run_in_graph_and_eager_modes
  def testUpdateRepeatedIndices2D(self):
    if test_util.is_gpu_available():
      self.skipTest("Duplicate indices scatter is non-deterministic on GPU")
    a = array_ops.zeros([10, 10])
    b = array_ops.tensor_scatter_update(
        a, [[5], [6], [6]],
        [math_ops.range(10),
         math_ops.range(11, 21),
         math_ops.range(10, 20)])
    self.assertAllEqual(
        b[6],
        constant_op.constant([10., 11., 12., 13., 14., 15., 16., 17., 18.,
                              19.]))


class ScatterNdTensorDeterminismTest(ScatterNdTensorTest):

  def setUp(self):
    super().setUp()
    config.enable_op_determinism()

  def tearDown(self):
    super().tearDown()
    config.disable_op_determinism()

  def testDeterminism(self):
    a = array_ops.zeros([1])
    indices = array_ops.zeros([100000, 1], dtypes.int32)
    values = np.random.randn(100000)
    val = self.evaluate(array_ops.tensor_scatter_update(a, indices, values))
    for _ in range(5):
      val2 = self.evaluate(array_ops.tensor_scatter_update(a, indices, values))
      self.assertAllEqual(val, val2)


if __name__ == "__main__":
  test.main()
