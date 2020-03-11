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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


GRADIENT_TESTS_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64)


def _AsType(v, vtype):
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _FlatInnerDims(tensor, ndims=2):
  shape = list(tensor.shape)
  return tensor.reshape([
      functools.reduce(lambda x, y: x * y, shape[:-ndims + 1], 1)
  ] + shape[-ndims + 1:])


def _FlatOuterDims(tensor, ndims=2):
  shape = list(tensor.shape)
  return tensor.reshape(shape[:ndims - 1] + [
      functools.reduce(lambda x, y: x * y, shape[ndims - 1:], 1)
  ])


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
    with self.cached_session(use_gpu=True):
      for ref_shape, indices_shape in zip(ref_shapes, indices_shapes):
        num_updates = indices_shape[0]
        ixdim = indices_shape[-1]

        indexable_area_shape = ()
        for i in range(ixdim):
          indexable_area_shape += (ref_shape[i],)
        all_indices = [
            list(coord)
            for coord, _ in np.ndenumerate(
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
        ref_var.initializer.run()
        tf_scatter(ref_var, indices, updates).eval()

        # Compare
        self.assertAllClose(new, self.evaluate(ref_var))

  def _VariableRankTests(self, np_scatter, tf_scatter):
    for vtype in (np.int32, np.float16, np.float32, np.float64, np.complex64,
                  np.complex128):
      for itype in (np.int32, np.int64):
        self._VariableRankTest(np_scatter, tf_scatter, vtype, itype)

  def testSimple(self):
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
    ref = variables.Variable([0, 0, 0, 0, 0, 0, 0, 0], dtype=dtypes.float32)
    expected = np.array([0, 11, 0, 10, 9, 0, 0, 12])
    scatter = state_ops.scatter_nd_update(ref, indices, updates)
    init = variables.global_variables_initializer()

    with self.session(use_gpu=True) as sess:
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
    self.assertAllEqual(self.evaluate(update),
                        [b"qq", b"cc", b"ee", b"dd", b"aa", b"", b"", b"bb"])

  @test_util.run_deprecated_v1
  def testSimpleResource(self):
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
    ref = resource_variable_ops.ResourceVariable(
        [0, 0, 0, 0, 0, 0, 0, 0], dtype=dtypes.float32)
    expected = np.array([0, 11, 0, 10, 9, 0, 0, 12])
    scatter = state_ops.scatter_nd_update(ref, indices, updates)
    init = variables.global_variables_initializer()

    with self.session(use_gpu=True) as sess:
      self.evaluate(init)
      self.evaluate(scatter)
      self.assertAllClose(ref.eval(), expected)

  def testSimple2(self):
    indices = constant_op.constant([[1, 0], [1, 1]], dtype=dtypes.int32)
    updates = constant_op.constant([11., 12.], dtype=dtypes.float32)
    ref = variables.Variable(
        [[0., 0.], [0., 0.], [0., 0.]], dtype=dtypes.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]])
    scatter = state_ops.scatter_nd_update(ref, indices, updates)
    init = variables.global_variables_initializer()

    with self.session(use_gpu=True) as sess:
      self.evaluate(init)
      result = self.evaluate(scatter)
      self.assertAllClose(result, expected)

  def testSimple3(self):
    indices = constant_op.constant([[1]], dtype=dtypes.int32)
    updates = constant_op.constant([[11., 12.]], dtype=dtypes.float32)
    ref = variables.Variable(
        [[0., 0.], [0., 0.], [0., 0.]], dtype=dtypes.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]])
    scatter = state_ops.scatter_nd_update(ref, indices, updates)
    init = variables.global_variables_initializer()

    with self.session(use_gpu=True) as sess:
      self.evaluate(init)
      result = self.evaluate(scatter)
      self.assertAllClose(result, expected)

  @test_util.run_deprecated_v1
  def testVariableRankUpdate(self):
    self._VariableRankTests(_NumpyUpdate, state_ops.scatter_nd_update)

  @test_util.run_deprecated_v1
  def testVariableRankAdd(self):
    self._VariableRankTests(_NumpyAdd, state_ops.scatter_nd_add)

  @test_util.run_deprecated_v1
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

  @test_util.run_v1_only("b/120545219")
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
  #     var.initializer.run()
  #     session.run([update0, update1])
  #     self.assertAllEqual([False, True], self.evaluate(var))

  @test_util.run_v1_only("b/120545219")
  def testScatterOutOfRangeCpu(self):
    # TODO(simister): Re-enable once binary size increase due to
    # scatter_nd ops is under control.
    #  tf.scatter_nd_mul, tf.scatter_nd_div,
    for op in (state_ops.scatter_nd_add, state_ops.scatter_nd_sub,
               state_ops.scatter_nd_update):
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      with self.cached_session(use_gpu=False):
        ref = variables.VariableV1(params)
        ref.initializer.run()

        # Indices all in range, no problem.
        indices = np.array([[2], [0], [5]])
        op(ref, indices, updates).eval()

        # Test some out of range errors.
        indices = np.array([[-1], [0], [5]])
        with self.assertRaisesOpError(
            r"indices\[0\] = \[-1\] does not index into shape \[6\]"):
          op(ref, indices, updates).eval()

        indices = np.array([[2], [0], [6]])
        with self.assertRaisesOpError(
            r"indices\[2\] = \[6\] does not index into shape \[6\]"):
          op(ref, indices, updates).eval()

  def testRank3ValidShape(self):
    indices = array_ops.zeros([2, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    self.assertAllEqual(
        state_ops.scatter_nd_update(ref, indices,
                                    updates).get_shape().as_list(), shape)

  @test_util.run_v1_only("b/120545219")
  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testResVarInvalidOutputShape(self):
    res = variables.Variable(
        initial_value=lambda: array_ops.zeros(shape=[], dtype=dtypes.float32),
        dtype=dtypes.float32)
    with self.cached_session():
      res.initializer.run()
      with self.assertRaisesOpError("Output must be at least 1-D"):
        state_ops.scatter_nd_update(res, [[0]], [0.22]).eval()

  @test_util.run_deprecated_v1
  def testExtraIndicesDimensions(self):
    indices = array_ops.zeros([1, 1, 2], dtypes.int32)
    updates = array_ops.zeros([1, 1], dtypes.int32)
    shape = np.array([2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    scatter_update = state_ops.scatter_nd_update(ref, indices, updates)
    self.assertAllEqual(scatter_update.get_shape().as_list(), shape)

    expected_result = np.zeros([2, 2], dtype=np.int32)
    with self.cached_session():
      ref.initializer.run()
      self.assertAllEqual(expected_result, self.evaluate(scatter_update))

  @test_util.run_deprecated_v1
  def testRank3InvalidShape1(self):
    indices = array_ops.zeros([3, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"The outer \d+ dimensions of indices\.shape="):
      state_ops.scatter_nd_update(ref, indices, updates)

  @test_util.run_deprecated_v1
  def testRank3InvalidShape2(self):
    indices = array_ops.zeros([2, 2, 1], dtypes.int32)
    updates = array_ops.zeros([2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"The inner \d+ dimensions of input\.shape="):
      state_ops.scatter_nd_update(ref, indices, updates)

  @test_util.run_deprecated_v1
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

    with session.Session() as sess:
      self.evaluate(init)
      result = self.evaluate(scatter)
      assert np.allclose(result, expected_result)

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
        ref.initializer.run()

        # Indices all in range, no problem.
        indices = np.array([2, 0, 5])
        op(ref, indices, updates).eval()

        # Indices out of range should not fail.
        indices = np.array([-1, 0, 5])
        op(ref, indices, updates).eval()
        indices = np.array([2, 0, 6])
        op(ref, indices, updates).eval()


class ScatterNdTest(test.TestCase):
  non_aliasing_add_test = False

  def scatter_nd(self, indices, updates, shape, input_=None):
    del input_  # input_ is not used in scatter_nd
    return array_ops.scatter_nd(indices, updates, shape)

  @test_util.run_in_graph_and_eager_modes
  def testBool(self):
    indices = constant_op.constant(
        [[4], [3], [1], [7]], dtype=dtypes.int32)
    updates = constant_op.constant(
        [False, True, False, True], dtype=dtypes.bool)
    expected = np.array(
        [False, False, False, True, False, False, False, True])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    result = self.evaluate(scatter)
    self.assertAllEqual(expected, result)

    # Same indice is updated twice by same value.
    indices = constant_op.constant(
        [[4], [3], [3], [7]], dtype=dtypes.int32)
    updates = constant_op.constant(
        [False, True, True, True], dtype=dtypes.bool)
    expected = np.array([
        False, False, False, True, False, False, False, True])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    result = self.evaluate(scatter)
    self.assertAllEqual(expected, result)

  @test_util.run_in_graph_and_eager_modes
  def testInvalidShape(self):
    # TODO(apassos) figure out how to unify these errors
    with self.assertRaises(errors.InvalidArgumentError
                           if context.executing_eagerly() else ValueError):
      array_ops.scatter_nd(indices=[0],  # this should be indices=[[0]]
                           updates=[0.0],
                           shape=[1])

  def testString(self):
    indices = constant_op.constant([[4], [3], [1], [7]],
                                   dtype=dtypes.int32)
    updates = constant_op.constant(["four", "three", "one", "seven"],
                                   dtype=dtypes.string)
    expected = np.array([b"", b"one", b"", b"three", b"four",
                         b"", b"", b"seven"])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    with self.cached_session() as sess:
      result = self.evaluate(scatter)
      self.assertAllEqual(expected, result)

    # Same indice is updated twice by same value.
    indices = constant_op.constant([[4], [3], [3], [7]],
                                   dtype=dtypes.int32)
    updates = constant_op.constant(["a", "b", "b", "c"],
                                   dtype=dtypes.string)
    expected = np.array([b"", b"", b"", b"bb", b"a", b"", b"", b"c"])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    with self.cached_session() as sess:
      result = self.evaluate(scatter)
      self.assertAllEqual(expected, result)

    # Same indice is updated twice by different value.
    indices = constant_op.constant([[4], [3], [3], [7]],
                                   dtype=dtypes.int32)
    updates = constant_op.constant(["a", "b", "c", "d"],
                                   dtype=dtypes.string)
    expected = [np.array([b"", b"", b"", b"bc", b"a", b"", b"", b"d"]),
                np.array([b"", b"", b"", b"cb", b"a", b"", b"", b"d"])]
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    with self.cached_session() as sess:
      result = self.evaluate(scatter)
      self.assertTrue(np.array_equal(result, expected[0]) or
                      np.array_equal(result, expected[1]))

  def testRank3ValidShape(self):
    indices = array_ops.zeros([2, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    self.assertAllEqual(
        self.scatter_nd(indices, updates, shape).get_shape().as_list(), shape)

  @test_util.run_deprecated_v1
  def testExtraIndicesDimensions(self):
    indices = array_ops.zeros([1, 1, 2], dtypes.int32)
    updates = array_ops.zeros([1, 1], dtypes.int32)
    shape = np.array([2, 2])
    scatter = self.scatter_nd(indices, updates, shape)
    self.assertAllEqual(scatter.get_shape().as_list(), shape)
    expected_result = np.zeros([2, 2], dtype=np.int32)
    with self.cached_session():
      self.assertAllEqual(expected_result, self.evaluate(scatter))

  @test_util.run_deprecated_v1
  def testUndefinedIndicesShape(self):
    indices = array_ops.placeholder(dtypes.int32, shape=None)
    updates = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
    shape = constant_op.constant([2, 2, 2], dtypes.int32)
    self.scatter_nd(indices, updates, shape)

  @test_util.run_deprecated_v1
  def testUndefinedUpdatesShape(self):
    indices = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
    updates = array_ops.placeholder(dtypes.int32, shape=None)
    shape = constant_op.constant([2, 2, 2], dtypes.int32)
    self.scatter_nd(indices, updates, shape)

  @test_util.run_deprecated_v1
  def testUndefinedOutputShape(self):
    indices = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
    updates = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
    shape = array_ops.placeholder(dtypes.int32, shape=[None])
    self.scatter_nd(indices, updates, shape)

  @test_util.run_deprecated_v1
  def testEmptyOutputShape1(self):
    indices = array_ops.zeros([2, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = constant_op.constant([0, 3, 2], dtypes.int32)

    with self.assertRaisesWithPredicateMatch(
        ValueError, "Indices and updates specified for empty output shape"):
      self.scatter_nd(indices, updates, shape)

  @test_util.run_v1_only("b/120545219")
  def testEmptyOutputShape2(self):
    indices = array_ops.placeholder(dtypes.int32, shape=None)
    updates = array_ops.placeholder(dtypes.int32, shape=None)
    shape = constant_op.constant([0, 3, 2], dtypes.int32)

    with self.cached_session():
      with self.assertRaisesOpError(
          "Indices and updates specified for empty output"):
        self.scatter_nd(indices, updates, shape).eval(feed_dict={
            indices: np.zeros([2, 2, 2], dtype=np.int32),
            updates: np.zeros([2, 2, 2], dtype=np.int32)
        })

  @test_util.run_deprecated_v1
  def testEmptyOutputShape3(self):
    indices = array_ops.zeros([0], dtypes.int32)
    updates = array_ops.zeros([0], dtypes.int32)
    shape = constant_op.constant([0], dtypes.int32)
    scatter = self.scatter_nd(indices, updates, shape)

    with self.cached_session():
      self.assertEqual(scatter.eval().size, 0)

  @test_util.run_deprecated_v1
  def testRank3InvalidShape1(self):
    indices = array_ops.zeros([3, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"The outer \d+ dimensions of indices\.shape="):
      self.scatter_nd(indices, updates, shape)

  @test_util.run_deprecated_v1
  def testRank3InvalidShape2(self):
    indices = array_ops.zeros([2, 2, 1], dtypes.int32)
    updates = array_ops.zeros([2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"The inner \d+ dimensions of (input|output)\.shape="):
      self.scatter_nd(indices, updates, shape)

  @test_util.run_deprecated_v1
  def testGradientsRank2ElementUpdate(self):
    for dtype in GRADIENT_TESTS_DTYPES:
      indices = constant_op.constant([[0, 0], [1, 1]], dtype=dtypes.int32)
      updates = constant_op.constant([1, 4], dtype=dtype)
      shape = constant_op.constant([2, 2], dtype=dtypes.int32)
      input_ = array_ops.zeros(shape, dtype=dtype)
      outputs = self.scatter_nd(indices, updates, shape, input_)

      grad_vals = constant_op.constant([[1, 2], [3, 4]], dtype=dtype)
      updates_grad, input_grad = gradients_impl.gradients(
          [outputs], [updates, input_], [grad_vals])
      expected_updates_grad = np.array([1, 4], dtype=dtype.as_numpy_dtype())
      expected_input_grad = np.array([[1, 2], [3, 4]],
                                     dtype=dtype.as_numpy_dtype())
      with self.cached_session():
        self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
        if self.non_aliasing_add_test:
          self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @test_util.run_deprecated_v1
  def testGradientsRank2SliceUpdate(self):
    for dtype in GRADIENT_TESTS_DTYPES:
      indices = constant_op.constant([[1], [0]], dtype=dtypes.int32)
      updates = constant_op.constant([[3, 4], [1, 2]], dtype=dtype)
      shape = constant_op.constant([2, 2], dtype=dtypes.int32)
      input_ = array_ops.zeros(shape, dtype=dtype)
      outputs = self.scatter_nd(indices, updates, shape, input_)

      grad_vals = constant_op.constant([[3, 4], [1, 2]], dtype=dtype)
      updates_grad, input_grad = gradients_impl.gradients(
          [outputs], [updates, input_], [grad_vals])
      expected_updates_grad = np.array([[1, 2], [3, 4]],
                                       dtype=dtype.as_numpy_dtype())
      expected_input_grad = np.array([[3, 4], [1, 2]],
                                     dtype=dtype.as_numpy_dtype())
      with self.cached_session():
        self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
        if self.non_aliasing_add_test:
          self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @test_util.run_deprecated_v1
  def testGradientsRank3SliceUpdate(self):
    for dtype in GRADIENT_TESTS_DTYPES:
      indices = constant_op.constant([[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                                     dtype=dtypes.int32)
      updates = constant_op.constant([[[5, 7], [2, 4]], [[1, 3], [6, 8]]],
                                     dtype=dtype)
      shape = constant_op.constant([2, 2, 2], dtype=dtypes.int32)
      input_ = array_ops.zeros(shape, dtype=dtype)
      outputs = self.scatter_nd(indices, updates, shape, input_)

      grad_vals = constant_op.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                                       dtype=dtype)
      updates_grad, input_grad = gradients_impl.gradients(
          [outputs], [updates, input_], [grad_vals])
      expected_updates_grad = np.array([[[3, 4], [5, 6]], [[1, 2], [7, 8]]],
                                       dtype=dtype.as_numpy_dtype())
      expected_input_grad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                                     dtype=dtype.as_numpy_dtype())
      with self.cached_session():
        self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
        if self.non_aliasing_add_test:
          self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @test_util.run_deprecated_v1
  def testGradientsRank7SliceUpdate(self):
    for dtype in GRADIENT_TESTS_DTYPES:
      indices = constant_op.constant(
          [[[[[[[0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]]]],
             [[[[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1]]]]]]],
          dtype=dtypes.int32)
      updates = constant_op.constant(
          [[[[[[[5, 6], [2, 4]]]], [[[[1, 3], [6, 8]]]]]]], dtype=dtype)
      shape = constant_op.constant([1, 1, 2, 1, 1, 2, 2], dtype=dtypes.int32)
      input_ = array_ops.zeros(shape, dtype=dtype)
      outputs = self.scatter_nd(indices, updates, shape, input_)

      grad_vals = constant_op.constant(
          [[[[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]]]], dtype=dtype)
      updates_grad, input_grad = gradients_impl.gradients(
          [outputs], [updates, input_], [grad_vals])
      expected_updates_grad = np.array(
          [[[[[[[3, 4], [5, 6]]]], [[[[1, 2], [7, 8]]]]]]],
          dtype=dtype.as_numpy_dtype())
      expected_input_grad = np.array(
          [[[[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]]]],
          dtype=dtype.as_numpy_dtype())
      with self.cached_session():
        self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
        if self.non_aliasing_add_test:
          self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @test_util.run_deprecated_v1
  def testScatterNdRepeatedIndicesAdd(self):
    indices = array_ops.zeros([100000, 1], dtypes.int32)
    values = np.random.randn(100000)
    shape = [1]
    with self.cached_session():
      val = self.scatter_nd(indices, values, shape).eval()
    self.assertAllClose([np.sum(values)], val)

  @test_util.run_deprecated_v1
  def testSmokeScatterNdBatch2DSliceDim2(self):
    with self.cached_session():
      indices = array_ops.zeros([3, 5, 2], dtype=dtypes.int32)
      values = array_ops.zeros([3, 5, 7])
      shape = [4, 6, 7]
      self.scatter_nd(indices, values, shape).eval()

  @test_util.run_deprecated_v1
  def testSmokeScatterNdBatch1DSliceDim2(self):
    with self.cached_session():
      indices = array_ops.zeros([0, 2], dtype=dtypes.int32)
      values = array_ops.zeros([0, 7])
      shape = [4, 6, 7]
      self.scatter_nd(indices, values, shape).eval()

  @test_util.run_deprecated_v1
  def testSmokeScatterNdBatch1DSliceDim3ShapeRank7(self):
    with self.cached_session():
      indices = array_ops.zeros([1, 3], dtype=dtypes.int32)
      values = array_ops.zeros([1, 6, 7, 8, 9])
      shape = [3, 4, 5, 6, 7, 8, 9]
      self.scatter_nd(indices, values, shape).eval()

  @test_util.run_deprecated_v1
  def testSmokeScatterNdBatch2DSliceDim3ShapeRank7(self):
    with self.cached_session():
      indices = array_ops.zeros([1, 2, 3], dtype=dtypes.int32)
      values = array_ops.zeros([1, 2, 6, 7, 8, 9])
      shape = [3, 4, 5, 6, 7, 8, 9]
      self.scatter_nd(indices, values, shape).eval()


class ScatterNdNonAliasingAddTest(ScatterNdTest):
  non_aliasing_add_test = True

  def scatter_nd(self, indices, updates, shape, input_=None):
    input_ = (input_ if input_ is not None else array_ops.zeros(
        shape, dtype=updates.dtype))
    return array_ops.scatter_nd_non_aliasing_add(input_, indices, updates)

  def testString(self):
    # Not supported yet.
    pass


class ScatterNdTensorTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testUpdateAddSub(self):
    indices = constant_op.constant([[4], [3], [1], [7]])
    updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
    t = array_ops.ones([8], dtype=dtypes.float32)
    assigned = array_ops.tensor_scatter_update(t, indices, updates)
    added = array_ops.tensor_scatter_add(t, indices, updates)
    subbed = array_ops.tensor_scatter_sub(t, indices, updates)

    self.assertAllEqual(assigned,
                        constant_op.constant([1, 11, 1, 10, 9, 1, 1, 12]))
    self.assertAllEqual(added,
                        constant_op.constant([1, 12, 1, 11, 10, 1, 1, 13]))
    self.assertAllEqual(subbed,
                        constant_op.constant([1, -10, 1, -9, -8, 1, 1, -11]))

  @test_util.run_v1_only("b/120545219")
  def testUpdateAddSubGradients(self):

    with self.cached_session():
      indices = constant_op.constant([[3], [1]])
      updates = constant_op.constant([9, 10], dtype=dtypes.float32)
      x = array_ops.ones([4], dtype=dtypes.float32)

      assigned = array_ops.tensor_scatter_update(x, indices, updates)
      added = array_ops.tensor_scatter_add(x, indices, updates)
      subbed = array_ops.tensor_scatter_sub(x, indices, updates)

      err_assigned = gradient_checker.compute_gradient_error(
          x, [4], assigned, [4])
      err_added = gradient_checker.compute_gradient_error(x, [4], added, [4])
      err_subbed = gradient_checker.compute_gradient_error(x, [4], subbed, [4])

      self.assertLess(err_assigned, 2e-4)
      self.assertLess(err_added, 2e-4)
      self.assertLess(err_subbed, 2e-4)

      err_assigned_wrt_updates = gradient_checker.compute_gradient_error(
          updates, [2], assigned, [4])
      err_added_wrt_updates = gradient_checker.compute_gradient_error(
          updates, [2], added, [4])
      err_subbed_wrt_updates = gradient_checker.compute_gradient_error(
          updates, [2], subbed, [4])

      self.assertLess(err_assigned_wrt_updates, 2e-4)
      self.assertLess(err_added_wrt_updates, 2e-4)
      self.assertLess(err_subbed_wrt_updates, 2e-4)

  def testTensorScatterUpdateWithForwarding(self):
    @def_function.function
    def _TestFn():
      indices = constant_op.constant([[4], [3], [1], [7]])
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      t = array_ops.ones([8], dtype=dtypes.float32)

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


if __name__ == "__main__":
  test.main()
