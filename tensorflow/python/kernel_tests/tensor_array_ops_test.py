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
"""Tests for tensorflow.ops.tensor_array_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_grad
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def _make_converter(tf_dtype):
  def _converter(x):
    if tf_dtype == dtypes.string:
      # In Python3, np.str is unicode, while we always want bytes
      return np.asarray(x).astype("|S")
    x = np.asarray(x).astype(tf_dtype.as_numpy_dtype)
    if tf_dtype.is_complex:
      # Add a non-zero imaginary component to x.
      x -= 1j * x
    return x
  return _converter


def _make_ta(size, name, dtype=dtypes.float32, infer_shape=False):
  return tensor_array_ops.TensorArray(
      dtype=dtype, tensor_array_name=name, size=size, infer_shape=infer_shape)


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class TensorArrayTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TensorArrayTest, cls).setUpClass()
    cls._workers, _ = test.create_local_cluster(num_workers=3, num_ps=0)

  @classmethod
  def tearDownClass(cls):
    super(TensorArrayTest, cls).tearDownClass()
    session_lib.Session.reset(cls._workers[0].target)

  @test_util.run_in_graph_and_eager_modes
  def testTensorArrayWriteRead(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      d0, d1, d2 = self.evaluate([r0, r1, r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)

  def _testTensorArrayWritePack(self, tf_dtype):
    with self.cached_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      convert = _make_converter(tf_dtype)

      w0 = ta.write(0, convert([[4.0, 5.0]]))
      w1 = w0.write(1, convert([[6.0, 7.0]]))
      w2 = w1.write(2, convert([[8.0, 9.0]]))

      c0 = w2.stack()

      c0 = self.evaluate(c0)
      self.assertAllEqual(
          convert([[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]), c0)

  def _testTensorArrayWritePackMaybeLegacy(self):
    self._testTensorArrayWritePack(dtypes.float32)
    self._testTensorArrayWritePack(dtypes.float64)
    self._testTensorArrayWritePack(dtypes.int32)
    self._testTensorArrayWritePack(dtypes.int64)
    self._testTensorArrayWritePack(dtypes.complex64)
    self._testTensorArrayWritePack(dtypes.complex128)
    self._testTensorArrayWritePack(dtypes.string)

  def testTensorArrayWritePack(self):
    self._testTensorArrayWritePackMaybeLegacy()

  def testEmptyTensorArrayPack(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)

      empty_element = np.zeros((0, 1), dtype=np.float32)
      w0 = ta.write(0, empty_element)
      w1 = w0.write(1, empty_element)
      w2 = w1.write(2, empty_element)

      c0 = w2.stack()

      c0 = self.evaluate(c0)
      self.assertAllEqual([3, 0, 1], c0.shape)

  def _testTensorArrayWriteConcat(self, tf_dtype):
    with self.cached_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3, infer_shape=False)

      convert = _make_converter(tf_dtype)

      w0 = ta.write(0, convert([[4.0, 5.0], [104.0, 105.0], [204.0, 205.0]]))
      w1 = w0.write(1, convert([[6.0, 7.0], [106.0, 107.0]]))
      w2 = w1.write(2, convert([[8.0, 9.0]]))

      c0 = w2.concat()

      c0 = self.evaluate(c0)
      self.assertAllEqual(
          convert([[4.0, 5.0], [104.0, 105.0], [204.0, 205.0], [6.0, 7.0],
                   [106.0, 107.0], [8.0, 9.0]]), c0)

  @test_util.deprecated_graph_mode_only
  def testTensorArrayWriteConcat(self):
    self._testTensorArrayWriteConcat(dtypes.float32)
    self._testTensorArrayWriteConcat(dtypes.float64)
    self._testTensorArrayWriteConcat(dtypes.int32)
    self._testTensorArrayWriteConcat(dtypes.int64)
    self._testTensorArrayWriteConcat(dtypes.complex64)
    self._testTensorArrayWriteConcat(dtypes.complex128)
    self._testTensorArrayWriteConcat(dtypes.string)

  def _testTensorArrayReadOrPackNotAllValuesAvailableFillsZeros(self):
    with self.cached_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          element_shape=tensor_shape.TensorShape([1, 2]))
      self.assertAllEqual([[0.0, 0.0]], self.evaluate(ta.read(0)))
      self.assertAllEqual([[[0.0, 0.0]], [[4.0, 5.0]], [[0.0, 0.0]]],
                          self.evaluate(ta.write(1, [[4.0, 5.0]]).stack()))
      self.assertAllEqual([[0.0, 0.0], [4.0, 5.0], [0.0, 0.0]],
                          self.evaluate(ta.write(1, [[4.0, 5.0]]).concat()))

  @test_util.run_v1_only("b/122324791")
  def testTensorArrayReadOrPackNotAllValuesAvailableFillsZeros(self):
    self._testTensorArrayReadOrPackNotAllValuesAvailableFillsZeros()

  def _testTensorArrayReadOrPackNotAllValuesAvailableInferShapeFillsZeros(self):
    ta = tensor_array_ops.TensorArray(
        dtype=dtypes.float32,
        tensor_array_name="foo",
        size=3)
    self.assertAllEqual(
        [[0.0, 0.0]], self.evaluate(ta.write(1, [[4.0, 5.0]]).read(0)))
    self.assertAllEqual([[[0.0, 0.0]], [[4.0, 5.0]], [[0.0, 0.0]]],
                        self.evaluate(ta.write(1, [[4.0, 5.0]]).stack()))
    self.assertAllEqual([[0.0, 0.0], [4.0, 5.0], [0.0, 0.0]],
                        self.evaluate(ta.write(1, [[4.0, 5.0]]).concat()))

  @test_util.run_v1_only("b/122324791")
  def testTensorArrayReadOrPackNotAllValuesAvailableInferShapeFillsZeros(self):
    self._testTensorArrayReadOrPackNotAllValuesAvailableInferShapeFillsZeros()

  @test_util.run_v1_only("Uses placeholders")
  def testSkipEagerTensorArrayReadUninitializedInferShapeFillsZeros(self):
    with self.cached_session(use_gpu=True) as sess:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3)
      val = array_ops.placeholder(dtypes.float32)
      self.assertAllEqual(
          [[0.0, 0.0]], sess.run(ta.write(1, val).read(0), {val: [[4.0, 5.0]]}))

  def _testTensorArrayUnpackRead(self, tf_dtype):
    with self.cached_session(use_gpu=True):
      convert = _make_converter(tf_dtype)

      ta = _make_ta(3, "foo", dtype=tf_dtype)
      # Unpack a vector into scalars
      w0 = ta.unstack(convert([1.0, 2.0, 3.0]))
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = self.evaluate([r0, r1, r2])
      self.assertAllEqual(convert(1.0), d0)
      self.assertAllEqual(convert(2.0), d1)
      self.assertAllEqual(convert(3.0), d2)

      # Unpack a matrix into vectors
      w1 = ta.unstack(convert([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]))
      r0 = w1.read(0)
      r1 = w1.read(1)
      r2 = w1.read(2)

      d0, d1, d2 = self.evaluate([r0, r1, r2])
      self.assertAllEqual(convert([1.0, 1.1]), d0)
      self.assertAllEqual(convert([2.0, 2.1]), d1)
      self.assertAllEqual(convert([3.0, 3.1]), d2)

      # Try unpacking an empty matrix, which should not cause an error.
      w2 = ta.unstack(convert([[], [], []]))
      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      d0, d1, d2 = self.evaluate([r0, r1, r2])
      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

  def _testTensorArrayUnpackReadMaybeLegacy(self):
    self._testTensorArrayUnpackRead(dtypes.float32)
    self._testTensorArrayUnpackRead(dtypes.float64)
    self._testTensorArrayUnpackRead(dtypes.int32)
    self._testTensorArrayUnpackRead(dtypes.int64)
    self._testTensorArrayUnpackRead(dtypes.complex64)
    self._testTensorArrayUnpackRead(dtypes.complex128)
    self._testTensorArrayUnpackRead(dtypes.string)

  def testTensorArrayUnpackRead(self):
    self._testTensorArrayUnpackReadMaybeLegacy()

  def _testTensorArraySplitRead(self, tf_dtype):
    with self.cached_session(use_gpu=True):
      convert = _make_converter(tf_dtype)

      # Split an empty vector
      ta = _make_ta(3, "foo", dtype=tf_dtype)
      lengths = constant_op.constant([0, 0, 0])
      w0 = ta.split(convert([]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = self.evaluate([r0, r1, r2])
      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

      # Split a vector
      lengths = constant_op.constant([2, 0, 1])
      w0 = ta.split(convert([1.0, 2.0, 3.0]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = self.evaluate([r0, r1, r2])
      self.assertAllEqual(convert([1.0, 2.0]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([3.0]), d2)

      # Split a matrix
      lengths = constant_op.constant([2, 0, 1])
      w0 = ta.split(
          convert([[1.0, 101.0], [2.0, 201.0], [3.0, 301.0]]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = self.evaluate([r0, r1, r2])
      self.assertAllEqual(convert([[1.0, 101.0], [2.0, 201.0]]), d0)
      self.assertAllEqual(convert([]).reshape(0, 2), d1)
      self.assertAllEqual(convert([[3.0, 301.0]]), d2)

  @test_util.deprecated_graph_mode_only
  def testTensorArraySplitRead(self):
    self._testTensorArraySplitRead(dtypes.float32)
    self._testTensorArraySplitRead(dtypes.float64)
    self._testTensorArraySplitRead(dtypes.int32)
    self._testTensorArraySplitRead(dtypes.int64)
    self._testTensorArraySplitRead(dtypes.complex64)
    self._testTensorArraySplitRead(dtypes.complex128)
    self._testTensorArraySplitRead(dtypes.string)

  @test_util.disable_control_flow_v2("v2 does not support TensorArray.grad.")
  @test_util.run_v1_only("v2 does not support TensorArray.grad.")
  def testSkipEagerTensorGradArrayWriteRead(self):
    with self.session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)
      g_ta = ta.grad("grad")

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      g_w0 = g_ta.write(0, [[5.0, 6.0]])
      g_w1 = g_w0.write(1, [[2.0]])
      g_w2 = g_w1.write(2, -2.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      g_r0 = g_w2.read(0)
      g_r1 = g_w2.read(1)
      g_r2 = g_w2.read(2)

      d0, d1, d2, g_d0, g_d1, g_d2 = session.run([r0, r1, r2, g_r0, g_r1, g_r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)
      self.assertAllEqual([[5.0, 6.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual(-2.0, g_d2)

  @test_util.deprecated_graph_mode_only
  def testSkipEagerTensorArrayGradGrad(self):
    if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
      self.skipTest("Legacy TensorArray does not support double derivatives.")
    with self.test_session(use_gpu=True) as session:
      x = constant_op.constant(4.0)

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=1,
          infer_shape=False)
      w0 = ta.write(0, x)
      r0 = w0.read(0)
      y = r0 * r0

      g1 = gradients_impl.gradients(ys=[y], xs=[x])
      g2 = gradients_impl.gradients(ys=[g1], xs=[x])
      self.assertAllEqual([2.0], session.run(g2))

  @test_util.disable_control_flow_v2("v2 does not support TensorArray.grad.")
  @test_util.run_v1_only("v2 does not support TensorArray.grad.")
  def testSkipEagerTensorGradArrayDynamicWriteRead(self):
    with self.session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True,
          infer_shape=False)

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      g_ta = w2.grad("grad")  # Get gradient array here so we know the shape

      s = w2.size()
      g_s = g_ta.size()

      g_w0 = g_ta.write(0, [[5.0, 6.0]])
      g_w1 = g_w0.write(1, [[2.0]])
      g_w2 = g_w1.write(2, -2.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      g_r0 = g_w2.read(0)
      g_r1 = g_w2.read(1)
      g_r2 = g_w2.read(2)

      d0, d1, d2, g_d0, g_d1, g_d2, vs, g_vs = session.run(
          [r0, r1, r2, g_r0, g_r1, g_r2, s, g_s])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)
      self.assertAllEqual([[5.0, 6.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual(-2.0, g_d2)
      self.assertAllEqual(3, vs)
      self.assertAllEqual(3, g_vs)

  @test_util.disable_control_flow_v2("v2 does not support TensorArray.grad.")
  @test_util.run_v1_only("v2 does not support TensorArray.grad.")
  def testSkipEagerTensorGradAccessTwiceReceiveSameObject(self):
    with self.session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      g_ta_0 = ta.grad("grad")
      g_ta_1 = ta.grad("grad")

      with ops.control_dependencies([g_ta_0.write(0, [[4.0, 5.0]]).flow]):
        # Write with one gradient handle, read with another copy of it
        r1_0 = g_ta_1.read(0)

      t_g_ta_0, t_g_ta_1, d_r1_0 = session.run(
          [g_ta_0.handle.op, g_ta_1.handle.op, r1_0])
      self.assertAllEqual(t_g_ta_0, t_g_ta_1)
      self.assertAllEqual([[4.0, 5.0]], d_r1_0)

  def testTensorArrayWriteWrongIndexOrDataTypeFails(self):
    with self.session(use_gpu=True):
      ta = _make_ta(3, "foo", dtype=dtypes.float32)
      # TODO(b/129870929): Remove the last 2 checks (runtime checks) after
      # back back from preferred_dtype= to dtype= in convert_to_tensor.  Also
      # restrict error check to only TypeError.
      error_msg_regex = (
          "("
          "Expected float32, got 'wrong_type_scalar' of type 'str' instead."
          "|"
          "Cannot convert provided value to EagerTensor. Provided value: "
          "wrong_type_scalar Requested dtype: float"
          "|"
          "TensorArray dtype is float.* but Op is trying to write dtype string"
          "|"
          "Invalid data types; op elements string but list elements float"
          ")")
      with self.assertRaisesRegex((TypeError, errors.InvalidArgumentError),
                                  error_msg_regex):
        self.evaluate(ta.write(0, "wrong_type_scalar").flow)

      if (control_flow_util.ENABLE_CONTROL_FLOW_V2 and
          not context.executing_eagerly()):
        error_msg = "Trying to modify element -1 in a list with 3 elements."
      else:
        error_msg = "index -1"
      with self.assertRaisesOpError(error_msg):
        self.evaluate(ta.write(-1, 3.0).flow)

      if (control_flow_util.ENABLE_CONTROL_FLOW_V2 and
          not context.executing_eagerly()):
        error_msg = "Trying to modify element 3 in a list with 3 elements"
      else:
        error_msg = ("Tried to write to index 3 but array is not "
                     "resizeable and size is: 3")
      # Test reading from too large an index
      with self.assertRaisesOpError(error_msg):
        self.evaluate(ta.write(3, 3.0).flow)

  def testTensorArrayReadWrongIndexOrDataTypeFails(self):
    with self.session(use_gpu=True):
      ta = _make_ta(3, "foo", dtype=dtypes.float32)

      w0 = ta.write(0, [[4.0, 5.0]])

      # Test reading wrong datatype (only possible when constructing graphs).
      if (not context.executing_eagerly() and
          not control_flow_util.ENABLE_CONTROL_FLOW_V2):
        r0_bad = gen_data_flow_ops.tensor_array_read_v3(
            handle=w0.handle, index=0, dtype=dtypes.float64, flow_in=w0.flow)
        with self.assertRaisesOpError(
            "TensorArray dtype is float but Op requested dtype double."):
          self.evaluate(r0_bad)

      if (control_flow_util.ENABLE_CONTROL_FLOW_V2 and
          not context.executing_eagerly()):
        error_msg = "Trying to access element -1 in a list with 3 elements."
      else:
        error_msg = "index -1"
      # Test reading from a negative index, which is not allowed
      with self.assertRaisesOpError(error_msg):
        self.evaluate(ta.read(-1))

      if (control_flow_util.ENABLE_CONTROL_FLOW_V2 and
          not context.executing_eagerly()):
        error_msg = "Trying to access element 3 in a list with 3 elements."
      else:
        error_msg = "Tried to read from index 3 but array size is: 3"
      # Test reading from too large an index
      with self.assertRaisesOpError(error_msg):
        self.evaluate(ta.read(3))

  @test_util.disable_control_flow_v2("v2 allows multiple writes.")
  @test_util.run_v1_only("v2 allows multiple writes.")
  def testSkipEagerTensorArrayWriteMultipleFails(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          "Could not write to TensorArray index 2 because "
          "it has already been written to."):
        self.evaluate(ta.write(2, 3.0).write(2, 3.0).flow)

  def testTensorArrayConcatIncompatibleShapesFails(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      w1 = ta.write(0, 3.0)
      w2 = w1.write(1, 4.0)
      w3 = w2.write(2, [3.0])

      with self.assertRaisesOpError(
          "Concat saw a scalar shape at index 0 but requires at least vectors"):
        self.evaluate(w3.concat())

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      w1 = ta.write(0, [3.0])
      w2 = w1.write(1, [4.0])
      w3 = w2.write(2, [[3.0]])

      # The exact error messages differ between eager execution and graph
      # construction as the former bubbles up the error from array_op.concat.
      error_msg = ("Incompatible ranks"
                   if control_flow_util.ENABLE_CONTROL_FLOW_V2 and
                   not context.executing_eagerly() else "shape")
      with self.assertRaisesRegex(errors.InvalidArgumentError, error_msg):
        self.evaluate(w3.concat())

  def testTensorArraySplitIncompatibleShapesFails(self):
    with self.session(use_gpu=True):
      in_eager_mode = context.executing_eagerly()
      ta = _make_ta(3, "foo")
      with self.assertRaisesOpError(
          r"Expected lengths to be a vector, received shape: \[\]"):
        if in_eager_mode:
          self.evaluate(ta.split([1.0, 2.0, 3.0], 1))
        else:
          lengths = array_ops.placeholder(dtypes.int64)
          ta.split([1.0, 2.0, 3.0], lengths).flow.eval(feed_dict={lengths: 1})

      error_msg = ("Unused values in tensor. Length of tensor: 3 Values used: 1"
                   if control_flow_util.ENABLE_CONTROL_FLOW_V2 and
                   not in_eager_mode else
                   r"Expected sum of lengths to be equal to values.shape\[0\], "
                   r"but sum of lengths is 1 and value's shape is: \[3\]")
      with self.assertRaisesOpError(error_msg):
        self.evaluate(ta.split([1.0, 2.0, 3.0], [1]).flow)

      ta = _make_ta(1, "baz")
      if control_flow_util.ENABLE_CONTROL_FLOW_V2 and not in_eager_mode:
        with self.assertRaisesRegex(
            ValueError, "Shape must be at least rank 1 but is rank 0"):
          self.evaluate(ta.split(1.0, [1]).flow)
      else:
        with self.assertRaisesOpError(
            r"Expected value to be at least a vector, but received shape: \[\]"
        ):
          self.evaluate(ta.split(1.0, [1]).flow)

      if not control_flow_util.ENABLE_CONTROL_FLOW_V2 or in_eager_mode:
        ta = _make_ta(2, "buz")
        with self.assertRaisesOpError(
            r"TensorArray's size is not equal to the size of lengths "
            r"\(2 vs. 1\), and the TensorArray is not marked as "
            r"dynamically resizeable"):
          self.evaluate(ta.split([1.0], [1]).flow)

  def _testTensorArrayWriteGradientAddMultipleAdds(self, dtype):
    with self.cached_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtype, tensor_array_name="foo", size=3, infer_shape=False)
      ta_grad = ta.grad("grad")

      c = lambda x: np.asarray(x, dtype=dtype.as_numpy_dtype)

      w0 = ta.write(2, c(3.0))
      w1 = w0.write(2, c(4.0))

      w0_grad = ta_grad.write(2, c(3.0))
      w1_grad = w0_grad.write(2, c(4.0))
      w2_grad = w1_grad.write(2, c(5.0))

      # Assert that aggregation works correctly
      self.assertAllEqual(c(12.00), w2_grad.read(2))

      # Assert that if multiple_writes_aggregate is not enabled,
      # multiple writes raise an exception.
      with self.assertRaisesOpError(
          r"TensorArray foo_.*: Could not write to TensorArray index 2 because "
          r"it has already been written to."):
        w1.flow.eval()

      # Using differing shapes causes an exception
      wb0_grad = ta_grad.write(1, c(1.0))
      wb1_grad = wb0_grad.write(1, c([1.0]))

      with self.assertRaisesOpError(
          r"Could not aggregate to TensorArray index 1 because the "
          r"existing shape is \[\] but the new input shape is \[1\]"):
        wb1_grad.flow.eval()

  @test_util.disable_control_flow_v2("v2 does not support TensorArray.grad.")
  @test_util.run_v1_only("v2 does not support TensorArray.grad.")
  def testSkipEagerTensorArrayWriteGradientAddMultipleAdds(self):
    for dtype in (dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64,
                  dtypes.complex64, dtypes.complex128):
      self._testTensorArrayWriteGradientAddMultipleAdds(dtype)

  @test_util.disable_control_flow_v2("Low level legacy TA op test.")
  @test_util.run_v1_only("Low level legacy TA op test.")
  def testSkipEagerTensorArrayGradWithShapeKnownElementShape(self):
    with self.session(use_gpu=True) as sess:
      ta = tensor_array_ops.TensorArray(
          size=3,
          dtype=dtypes.float32,
          element_shape=tensor_shape.TensorShape([2, 3]))
      handle, flow = data_flow_ops.tensor_array_grad_with_shape(
          handle=ta.handle,
          flow_in=ta.flow,
          shape_to_prepend=tensor_shape.TensorShape([4, 5]),
          source="source")
      ta_grad = tensor_array_ops.TensorArray(
          dtypes.float32, handle=handle, flow=flow)
      value = array_ops.placeholder(dtypes.float32)
      ta_grad = ta_grad.write(0, value)
      read_value = ta_grad.read(0)

      # Make sure shape inference worked.
      self.assertAllEqual([None, None, 2, 3], read_value.shape.as_list())
      # Writing with wrong shape should not work.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Could not write to TensorArray"):
        fed_value = np.random.random([2, 3])
        sess.run(read_value, feed_dict={value: fed_value})
      # Writing with correct shape should work.
      fed_value = np.random.random([4, 5, 2, 3])
      self.assertAllClose(fed_value,
                          sess.run(read_value, feed_dict={value: fed_value}))

  @test_util.disable_control_flow_v2("Low level legacy TA op test.")
  @test_util.run_v1_only("Low level legacy TA op test.")
  def testSkipEagerTensorArrayGradWithShapeUnknownElementShape(self):
    with self.session(use_gpu=True) as sess:
      ta = tensor_array_ops.TensorArray(
          size=3, dtype=dtypes.float32,
          element_shape=None)  # Note that element_shape is unknown
      handle, flow = data_flow_ops.tensor_array_grad_with_shape(
          handle=ta.handle,
          flow_in=ta.flow,
          shape_to_prepend=tensor_shape.TensorShape([4, 5]),
          source="source")
      ta_grad = tensor_array_ops.TensorArray(
          dtypes.float32, handle=handle, flow=flow)
      value = array_ops.placeholder(dtypes.float32)
      ta_grad = ta_grad.write(0, value)
      read_value = ta_grad.read(0)

      # Make sure shape inference worked.
      self.assertIsNone(read_value.shape.ndims)
      # Write with some shape and check read value.
      fed_value = np.random.random([4, 5, 7])
      self.assertAllClose(fed_value,
                          sess.run(read_value, feed_dict={value: fed_value}))

  def testMultiTensorArray(self):
    with self.session(use_gpu=True):
      h1 = tensor_array_ops.TensorArray(
          size=1, dtype=dtypes.float32, tensor_array_name="foo")
      w1 = h1.write(0, 4.0)
      r1 = w1.read(0)

      h2 = tensor_array_ops.TensorArray(
          size=1, dtype=dtypes.float32, tensor_array_name="bar")

      w2 = h2.write(0, 5.0)
      r2 = w2.read(0)
      r = r1 + r2
      val = self.evaluate(r)
      self.assertAllClose(9.0, val)

  def _testTensorArrayGradientWriteReadType(self, dtype):
    with self.cached_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.as_dtype(dtype),
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      c = lambda x: np.array(x, dtype=dtype)

      value_0 = constant_op.constant(c([[4.0, 5.0]]))
      value_1 = constant_op.constant(c(3.0))

      w0 = ta.write(0, value_0)
      w1 = w0.write(1, value_1)
      r0 = w1.read(0)
      r1 = w1.read(1)
      r0_2 = w1.read(0)

      # Test individual components' gradients
      grad_just_r0 = gradients_impl.gradients(
          ys=[r0], xs=[value_0], grad_ys=[c([[2.0, 3.0]])])
      grad_just_r0_vals = session.run(grad_just_r0)
      self.assertAllEqual(c([[2.0, 3.0]]), grad_just_r0_vals[0])

      grad_r0_r0_2 = gradients_impl.gradients(
          ys=[r0, r0_2],
          xs=[value_0],
          grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]])])
      grad_r0_r0_2_vals = session.run(grad_r0_r0_2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_r0_r0_2_vals[0])

      grad_just_r1 = gradients_impl.gradients(
          ys=[r1], xs=[value_1], grad_ys=[c(-2.0)])
      grad_just_r1_vals = session.run(grad_just_r1)
      self.assertAllEqual(c(-2.0), grad_just_r1_vals[0])

      # Test combined gradients
      grad = gradients_impl.gradients(
          ys=[r0, r0_2, r1],
          xs=[value_0, value_1],
          grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]]), c(-2.0)])
      grad_vals = session.run(grad)
      self.assertEqual(len(grad_vals), 2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_vals[0])
      self.assertAllEqual(c(-2.0), grad_vals[1])

  @test_util.deprecated_graph_mode_only
  def testSkipEagerTensorArrayGradientWriteRead(self):
    for dtype in (np.float32, np.float64, np.complex64, np.complex128):
      self._testTensorArrayGradientWriteReadType(dtype)

  def _testTensorArrayGradientWritePackConcatAndRead(self):
    with self.cached_session(use_gpu=True) as sess:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=2,
          clear_after_read=False)

      value_0 = constant_op.constant([-1.0, 1.0])
      value_1 = constant_op.constant([-10.0, 10.0])

      w0 = ta.write(0, value_0)
      w1 = w0.write(1, value_1)
      p0 = w1.stack()
      r0 = w1.read(0)
      s0 = w1.concat()

      # Test gradient accumulation between read(0), pack(), and concat()
      with ops.control_dependencies([p0, r0, s0]):
        grad_r = gradients_impl.gradients(
            ys=[p0, r0, s0],
            xs=[value_0, value_1],
            grad_ys=[
                [[2.0, 3.0], [4.0, 5.0]],  # pack gradient
                [-0.5, 1.5],  # read(0) gradient
                [20.0, 30.0, 40.0, 50.0]
            ])  # concat gradient
      grad_vals = self.evaluate(grad_r)  # 2 + 2 entries

      self.assertAllClose([2.0 - 0.5 + 20.0, 3.0 + 1.5 + 30.0], grad_vals[0])
      self.assertAllEqual([4.0 + 40.0, 5.0 + 50.0], grad_vals[1])

  @test_util.deprecated_graph_mode_only
  def testSkipEagerTensorArrayGradientWritePackConcatAndRead(self):
    self._testTensorArrayGradientWritePackConcatAndRead()

  @test_util.disable_control_flow_v2("v2 does not support clear_after_read.")
  @test_util.run_v1_only("v2 does not support clear_after_read.")
  def testTensorArrayReadTwice(self):
    with self.session(use_gpu=True):
      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      ta_readonce = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=2)

      w_readonce = ta_readonce.unstack(value)
      r0_readonce = w_readonce.read(0)

      with self.assertRaisesOpError(
          r"Could not read index 0 twice because it was cleared after a "
          r"previous read \(perhaps try setting clear_after_read = false\?\)"):
        with ops.control_dependencies([r0_readonce]):
          self.evaluate(w_readonce.read(0))

      ta_readtwice = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=2,
          clear_after_read=False)
      w_readtwice = ta_readtwice.unstack(value)
      r0_readtwice = w_readtwice.read(0)
      with ops.control_dependencies([r0_readtwice]):
        r1_readtwice = w_readtwice.read(0)

      self.assertAllEqual([1.0, -1.0], self.evaluate(r1_readtwice))

  def _testTensorArrayGradientUnpackRead(self):
    with self.cached_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=2,
          clear_after_read=False)

      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.unstack(value)
      r0 = w.read(0)
      r0_1 = w.read(0)
      r1 = w.read(1)

      # Test combined gradients + aggregation of read(0)
      grad = gradients_impl.gradients(
          ys=[r0, r0_1, r1],
          xs=[value],
          grad_ys=[[2.0, 3.0], [-1.5, 1.5], [4.0, 5.0]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0 - 1.5, 3.0 + 1.5], [4.0, 5.0]], grad_vals[0])

  @test_util.deprecated_graph_mode_only
  def testSkipEagerTensorArrayGradientUnpackRead(self):
    self._testTensorArrayGradientUnpackRead()

  @test_util.deprecated_graph_mode_only
  def testSkipEagerTensorArrayGradientSplitConcat(self):
    with self.session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=2,
          infer_shape=False)

      value = constant_op.constant(
          [[1.0, -1.0], [10.0, -10.0], [100.0, -100.0]])

      w = ta.split(value, [2, 1])
      r = w.concat()

      # Test combined gradients
      grad = gradients_impl.gradients(
          ys=[r],
          xs=[value],
          grad_ys=[[[2.0, -2.0], [20.0, -20.0], [200.0, -200.0]]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0, -2.0], [20.0, -20.0], [200.0, -200.0]],
                          grad_vals[0])

  def _testTensorArrayGradientDynamicUnpackRead(self):
    with self.cached_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True)

      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.unstack(value)
      r0 = w.read(0)
      r1 = w.read(1)

      # Test combined gradients + aggregation of read(0)
      grad = gradients_impl.gradients(
          ys=[r0, r1], xs=[value], grad_ys=[[2.0, 3.0], [4.0, 5.0]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0, 3.0], [4.0, 5.0]], grad_vals[0])

  @test_util.deprecated_graph_mode_only
  def testSkipEagerTensorArrayGradientDynamicUnpackRead(self):
    self._testTensorArrayGradientDynamicUnpackRead()

  def testCloseTensorArray(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      self.evaluate(ta.close())

  def testSizeTensorArray(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      s = ta.size()
      self.assertAllEqual(3, self.evaluate(s))

  def testWriteCloseTensorArray(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)
      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [3.0])
      self.evaluate(w1.close())  # Expected to run without problems

  def _testWhileLoopWritePackGradients(self, dynamic_size, dtype):
    np_dtype = dtype.as_numpy_dtype
    with self.cached_session(use_gpu=True):
      def func(v0, state0, var):
        ta = tensor_array_ops.TensorArray(
            dtype=dtype,
            tensor_array_name="foo",
            size=0 if dynamic_size else 3,
            dynamic_size=dynamic_size)
        time_0 = array_ops.identity(0)

        def body(time, ta_t, state):
          sliced = array_ops.slice(
              v0, begin=array_ops.stack([time, 0]), size=[1, -1])
          sliced = array_ops.squeeze(sliced)
          out = sliced + var + state
          state += sliced
          ta_t = ta_t.write(time, out)
          return (time + 1, ta_t, state)

        (unused_0, h_final, unused_2) = control_flow_ops.while_loop(
            cond=lambda time, unused_1, unused_2: time < 3,
            body=body,
            loop_vars=(time_0, ta, state0),
            shape_invariants=(time_0.get_shape(), tensor_shape.unknown_shape(),
                              tensor_shape.unknown_shape()),
            parallel_iterations=3)
        vout = h_final.stack()
        return vout

      v0 = array_ops.identity(np.arange(3 * 5, dtype=np_dtype).reshape(3, 5))
      state0 = array_ops.identity(np.array([1] * 5, dtype=np_dtype))
      init_val = np.arange(100, 105, dtype=np_dtype)
      var = variable_scope.get_variable(
          "var",
          shape=init_val.shape,
          dtype=np_dtype,
          initializer=init_ops.constant_initializer(init_val))

      vout = func(v0, state0, var)
      grad_val = -np.arange(3 * 5, dtype=np_dtype).reshape(3, 5)
      if context.executing_eagerly():
        grad_fn = backprop.gradients_function(func)
        v0_grad, state0_grad, var_grad = grad_fn(v0, state0, var, dy=grad_val)
      else:
        v0_grad = gradients_impl.gradients([vout], [v0], [grad_val])[0]
        state0_grad = gradients_impl.gradients([vout], [state0], [grad_val])[0]
        var_grad = gradients_impl.gradients([vout], [var], [grad_val])[0]
        self.evaluate(variables.global_variables_initializer())

      state0_t, var_t, v0_t, vout_t, v0_grad_t, var_grad_t, state0_grad_t = (
          self.evaluate(
              ([state0, var, v0, vout, v0_grad, var_grad, state0_grad])))
      just_v0_grad_t = self.evaluate(v0_grad)

      # state = [ state0 | state0 + v0[0] | state0 + v0[0] + v0[1] ]
      # vout = [ v0[0] + var + state[0] |
      #          v0[1] + var + state[1] |
      #          v0[2] + var + state[2] ]
      #      = [ v0[0] + var + state0 |
      #          v0[1] + var + state0 + v0[0] |
      #          v0[2] + var + state0 + v0[0] + v0[1] ]
      #
      # d(vout[0])/d(v0) = [1 | 0 | 0 ]
      # d(vout[1])/d(v0) = [1 | 1 | 0 ]
      # d(vout[2])/d(v0) = [1 | 1 | 1 ]
      # d(vout)/d(var) = [1 | 1 | 1]
      # d(vout)/d(state0) = [ 1 | 1 | 1 ]

      state_per_time = np.array(
          [state0_t, state0_t + v0_t[0, :], state0_t + v0_t[0, :] + v0_t[1, :]])

      # Compare forward prop
      self.assertAllClose(v0_t + var_t + state_per_time, vout_t)

      # Compare backward prop
      expected_v0_grad_t = np.array([
          grad_val[0, :] + grad_val[1, :] + grad_val[2, :],
          grad_val[1, :] + grad_val[2, :], grad_val[2, :]
      ])

      self.assertAllEqual(expected_v0_grad_t, v0_grad_t)
      self.assertAllEqual(expected_v0_grad_t, just_v0_grad_t)
      self.assertAllClose(grad_val.sum(axis=0), var_grad_t)
      self.assertAllClose(grad_val.sum(axis=0), state0_grad_t)

  def testWhileLoopWritePackGradients(self):
    self._testWhileLoopWritePackGradients(
        dynamic_size=False, dtype=dtypes.float32)
    # TODO(ebrevdo): re-enable when While supports non-float32 gradients.
    # self._testWhileLoopWritePackGradients(
    #     dynamic_size=False, dtype=tf.int64)

  @test_util.run_deprecated_v1
  def testSkipEagerWhileLoopDynamicWritePackGradients(self):
    self._testWhileLoopWritePackGradients(
        dynamic_size=True, dtype=dtypes.float32)

  def testGradSerialTwoLoops(self):
    with self.session(use_gpu=True):
      def loop(x):
        num_steps = 100
        acc = tensor_array_ops.TensorArray(
            dtype=dtypes.float32,
            size=num_steps,
            clear_after_read=False,
            element_shape=tensor_shape.TensorShape([]))
        i = constant_op.constant(0, name="i")

        c = lambda i, acc: i < 5

        def b(i, acc):
          x1 = control_flow_ops.cond(
              math_ops.equal(i, 0), lambda: x,
              lambda: math_ops.multiply(acc.read(i - 1), 2.0))
          return i + 1, acc.write(i, x1)

        i1, acc1 = control_flow_ops.while_loop(c, b, [i, acc])

        z = constant_op.constant(0.0)

        def fn(i, acc):
          return i + 1, acc.write(i, z)

        _, acc2 = control_flow_ops.while_loop(lambda i, acc: i < num_steps, fn,
                                              [i1, acc1])

        r = acc2.stack()
        return r

      x = constant_op.constant(2.0, name="x")
      if context.executing_eagerly():
        grad = backprop.gradients_function(loop)(x)[0]
      else:
        grad = gradients_impl.gradients(loop(x), [x])[0]
      self.assertAllClose(31.0, self.evaluate(grad))

  def testShapeAfterWhileLoop(self):
    size = 10
    ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=size)
    _, ta = control_flow_ops.while_loop(
        lambda i, _: i < size,
        lambda i, ta: (i + 1, ta.write(i, [[0.]])), [0, ta],
        parallel_iterations=1)
    self.assertIsNotNone(ta.element_shape.dims)

  @test_util.deprecated_graph_mode_only
  def testSkipEagerSumOfTwoReadVariablesWithoutRepeatGrad(self):
    with self.session(use_gpu=True) as session:
      a = array_ops.identity(
          np.arange(
              3 * 5, dtype=np.float32).reshape(3, 5) + 1)
      b = array_ops.identity(
          np.arange(
              3 * 5, dtype=np.float32).reshape(3, 5) + 1 + 3 * 5)
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
      ta = ta.write(0, a, name="write_a")
      ta = ta.write(1, b, name="write_b")
      c = (
          ta.read(
              0, name="read_a_0") +  # a + b
          ta.read(
              1, name="read_b_0"))
      g0 = -(np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1)
      grad_a = gradients_impl.gradients([c], [a], [g0])[0]  # d(a+b)/da = 1
      grad_b = gradients_impl.gradients([c], [b], [g0])[0]  # d(a+b)/db = 1

      # Test gradients calculated individually
      grad_a_t, = session.run([grad_a])
      self.assertAllEqual(grad_a_t, g0)

      grad_b_t, = session.run([grad_b])
      self.assertAllEqual(grad_b_t, g0)

      # Test gradients calculated jointly
      joint_grad_a_t, joint_grad_b_t = session.run([grad_a, grad_b])
      self.assertAllEqual(joint_grad_a_t, g0)
      self.assertAllEqual(joint_grad_b_t, g0)

  def _grad_source_for_name(self, name):
    return tensor_array_grad._GetGradSource(constant_op.constant(0, name=name))

  @test_util.deprecated_graph_mode_only
  def testSkipEagerGetGradSource_Invalid(self):
    with self.assertRaises(ValueError):
      self._grad_source_for_name("")
    with self.assertRaises(ValueError):
      self._grad_source_for_name("foo")
    with self.assertRaises(ValueError):
      self._grad_source_for_name("foo/bar")

  @test_util.deprecated_graph_mode_only
  def testSkipEagerGetGradSource_NoEnclosingScope(self):
    self.assertEqual("gradients:0", self._grad_source_for_name("gradients"))
    self.assertEqual("gradients_0:0", self._grad_source_for_name("gradients_0"))
    self.assertEqual("gradients", self._grad_source_for_name("gradients/foo"))
    self.assertEqual("gradients_0",
                     self._grad_source_for_name("gradients_0/foo"))
    self.assertEqual("gradients",
                     self._grad_source_for_name("gradients/foo/bar"))
    self.assertEqual("gradients_0",
                     self._grad_source_for_name("gradients_0/foo/bar"))

  @test_util.deprecated_graph_mode_only
  def testSkipEagerGetGradSource_EnclosingScope(self):
    self.assertEqual("foo/gradients:0",
                     self._grad_source_for_name("foo/gradients"))
    self.assertEqual("foo/gradients_0:0",
                     self._grad_source_for_name("foo/gradients_0"))
    self.assertEqual("foo/gradients",
                     self._grad_source_for_name("foo/gradients/bar"))
    self.assertEqual("foo/gradients_0",
                     self._grad_source_for_name("foo/gradients_0/bar"))
    self.assertEqual("foo/bar/gradients",
                     self._grad_source_for_name("foo/bar/gradients/baz"))
    self.assertEqual("foo/bar/gradients_0",
                     self._grad_source_for_name("foo/bar/gradients_0/baz"))

  @test_util.deprecated_graph_mode_only
  def testSkipEagerGetGradSource_NestedUsesInnermost(self):
    self.assertEqual(
        "foo/gradients/bar/gradients_0",
        self._grad_source_for_name("foo/gradients/bar/gradients_0/baz"))

  @test_util.deprecated_graph_mode_only
  def testSkipEagerWriteShape(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c0 = constant_op.constant([4.0, 5.0])
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual(c0.get_shape(), r0.get_shape())

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c1 = constant_op.constant([6.0, 7.0])
      w1 = w0.write(1, c1)
      r0 = w1.read(0)
      r1 = w1.read(1)
      self.assertAllEqual(c0.get_shape(), r0.get_shape())
      self.assertAllEqual(c1.get_shape(), r1.get_shape())

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c2 = constant_op.constant([4.0, 5.0, 6.0])
      with self.assertRaises(ValueError):
        w0.write(0, c2)

  @test_util.deprecated_graph_mode_only
  def testSkipEagerPartlyUnknownShape(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=6)

      c0 = array_ops.placeholder(dtypes.float32, [None, None, None, 3])
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual([None, None, None, 3], r0.get_shape().as_list())

      c1 = array_ops.placeholder(dtypes.float32, [None, None, None, 3])
      w1 = w0.write(1, c1)
      r1 = w1.read(0)
      self.assertAllEqual([None, None, None, 3], r1.get_shape().as_list())

      # Writing less specific shape (doesn't change type.)
      c2 = array_ops.placeholder(dtypes.float32, [None, None, None, None])
      w2 = w1.write(2, c2)
      r2 = w2.read(0)
      self.assertAllEqual([None, None, None, 3], r2.get_shape().as_list())

      # Writing more specific shape in one dimension and less specific in
      # another.
      c3 = array_ops.placeholder(dtypes.float32, [None, None, 2, None])
      w3 = w2.write(3, c3)
      r3 = w3.read(0)
      self.assertAllEqual([None, None, 2, 3], r3.get_shape().as_list())

      # Writing partly defined shape using TensorArray.scatter.
      c4 = array_ops.placeholder(dtypes.float32, [2, None, 4, 2, 3])
      w4 = w3.scatter([4, 5], c4)
      r4 = w4.read(0)
      self.assertAllEqual([None, 4, 2, 3], r4.get_shape().as_list())

      # Writing fully defined shape using TensorArray.split.
      c5 = array_ops.placeholder(dtypes.float32, [10, 4, 2, 3])
      w5 = w4.split(c5, constant_op.constant([5, 5]))
      r5 = w5.read(0)
      self.assertAllEqual([5, 4, 2, 3], r5.get_shape().as_list())

  def _testUnpackShape(self):
    with self.cached_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True,
          infer_shape=True)
      value = constant_op.constant(
          [[1.0, -1.0], [10.0, -10.0], [100.0, -100.0]])
      w0 = ta.unstack(value)
      r0 = w0.read(0)
      self.assertAllEqual((2,), r0.get_shape())

      c1 = constant_op.constant([4.0, 5.0])
      w1 = w0.write(3, c1)

      if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
        # TensorArray v2 does not support clear_after_read.
        with self.assertRaisesOpError(
            r"Could not read index 0 twice because it was cleared after a "
            r"previous read \(perhaps try setting clear_after_read = false\?\)"
        ):
          with ops.control_dependencies([r0]):
            self.evaluate(w1.read(0))

      r1 = w1.read(1)
      self.assertAllEqual(c1.get_shape(), r1.shape)

      c2 = constant_op.constant([4.0, 5.0, 6.0])
      with self.assertRaises(ValueError):
        w1.write(4, c2)

  def testUnpackShape(self):
    self._testUnpackShape()

  @test_util.deprecated_graph_mode_only
  def testSplitShape(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True,
          infer_shape=True)
      value = constant_op.constant([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
      w0 = ta.split(value, [1, 1, 1])
      r0 = w0.read(0)
      self.assertAllEqual((1, 2), r0.get_shape())

      ta1 = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo1",
          size=0,
          dynamic_size=True,
          infer_shape=True)
      w0 = ta1.split(value, [1, 2])
      r0 = w0.read(0)
      if context.executing_eagerly():
        self.assertEqual((1, 2), r0.get_shape())
        self.assertEqual((2, 2), w0.read(1).get_shape())
      else:
        self.assertEqual(r0.get_shape().ndims, None)
        if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
          self.assertEqual(
              tensor_shape.TensorShape(
                  ta1.handle.op.get_attr("element_shape")).ndims, None)

  @test_util.deprecated_graph_mode_only
  def testSkipEagerWriteUnknownShape(self):
    with self.session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=True)
      c0 = array_ops.placeholder(dtypes.float32)
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual(r0.get_shape(), tensor_shape.unknown_shape())

  def _testGradientWhenNotAllComponentsRead(self):
    with self.cached_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
      x = constant_op.constant([2.0, 3.0])
      w = ta.unstack(x)
      r0 = w.read(0)
      # calculate (dr0/dx0, dr0/dx1).  since r0 = x0, gradients are (1, 0).
      grad_r0 = gradients_impl.gradients(ys=[r0], xs=[x], grad_ys=[1.0])
      grad_r0_vals = session.run(grad_r0)[0]
      self.assertAllEqual(grad_r0_vals, [1.0, 0.0])

  @test_util.deprecated_graph_mode_only
  def testSkipEagerGradientWhenNotAllComponentsRead(self):
    self._testGradientWhenNotAllComponentsRead()

  @test_util.deprecated_graph_mode_only
  def testSkipEagerWriteButNotAllComponentsReadGrad(self):
    with self.cached_session(use_gpu=True) as session:
      x0 = constant_op.constant(5.0)
      x1 = constant_op.constant(10.0)
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=2).write(0, x0).write(1, x1)
      r0 = ta.read(0)
      # calculate (dr0/dx0, dr0/dx1).  since r0 = x0, gradients are (1, 0).
      grad_r0_x1 = gradients_impl.gradients(ys=[r0], xs=[x0, x1], grad_ys=[1.0])
      grad_r0_x1_vals = session.run(grad_r0_x1)
      self.assertAllEqual(grad_r0_x1_vals, [1.0, 0.0])

  def _testTensorArrayUnpackDynamic(self):
    with self.cached_session(use_gpu=True) as sess:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=3, dynamic_size=True)
      x = constant_op.constant([1.0, 2.0, 3.0])
      w0 = ta.unstack(x)
      w1 = w0.write(3, 4.0)
      r = w1.stack()
      self.assertAllEqual(np.array([1.0, 2.0, 3.0, 4.0]), self.evaluate(r))
      grad = gradients_impl.gradients(ys=[r], xs=[x])
      self.assertAllEqual(np.array([1.0, 1.0, 1.0]), self.evaluate(grad)[0])

  @test_util.run_deprecated_v1
  def testSkipEagerTensorArrayUnpackDynamic(self):
    self._testTensorArrayUnpackDynamic()

  @test_util.run_deprecated_v1
  def testSkipEagerTensorArraySplitDynamic(self):
    with self.session(use_gpu=True) as sess:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=3, dynamic_size=True)
      x = constant_op.constant([1.0, 2.0, 3.0])
      w0 = ta.split(x, [1, 1, 1])
      w1 = w0.write(3, [4.0])
      r = w1.concat()
      self.assertAllEqual(np.array([1.0, 2.0, 3.0, 4.0]), self.evaluate(r))
      grad = gradients_impl.gradients(ys=[r], xs=[x])
      self.assertAllEqual(np.array([1.0, 1.0, 1.0]), self.evaluate(grad)[0])

  def testStackShape(self):

    @def_function.function
    def ta_stack():
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=3)
      x = constant_op.constant([1.0, 2.0, 3.0])
      ta = ta.write(0, x)
      t = ta.stack()
      self.assertEqual(t.shape.as_list(), [3, 3])
      return t

    ta_stack()

  def testReadShape(self):

    @def_function.function
    def ta_read():
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=3)
      x = constant_op.constant([1.0, 2.0, 3.0])
      ta = ta.write(0, x)
      t = ta.read(0)
      self.assertEqual(t.shape.as_list(), [3])
      return t

    ta_read()

  def testGatherShape(self):

    def ta_gather(indices):
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=3)
      x = constant_op.constant([1.0, 2.0, 3.0])
      ta = ta.write(0, x)
      t = ta.gather(indices)
      self.assertEqual(t.shape.as_list(), [first_dim, 3])
      return t

    # This propagates shape of `indices` when compiling ta_gather.
    ta_gather_with_known_indices_shape = def_function.function(ta_gather)
    first_dim = 1
    ta_gather_with_known_indices_shape([0])

    # Here were force the shape of `indices` to be [None] during ta_gather's
    # compilation.
    ta_gather_with_unknown_indices_shape = def_function.function(
        ta_gather,
        input_signature=[
            tensor_spec.TensorSpec(dtype=dtypes.int32, shape=[None])
        ])
    first_dim = None
    ta_gather_with_unknown_indices_shape([0])

  def _testTensorArrayEvalEmpty(self):
    with self.cached_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=0, dynamic_size=False, infer_shape=False)
      v2_msg = ("Tried to stack elements of an empty list with "
                "non-fully-defined element_shape")
      v1_msg = (
          "TensorArray has size zero, but element shape <unknown> is not "
          "fully defined. Currently only static shapes are supported when "
          "packing zero-size TensorArrays.")
      with self.assertRaisesOpError(
          v2_msg if control_flow_util.ENABLE_CONTROL_FLOW_V2 else v1_msg):
        ta.stack().eval()

  @test_util.run_deprecated_v1
  def testSkipEagerTensorArrayEvalEmpty(self):
    self._testTensorArrayEvalEmpty()

  # this test is ill-defined for Eager mode --- unpacking an empty tensor
  # gives an empty list / there is not equivalent of "mark_used" in Eager
  def _testTensorArrayEvalEmptyWithDefault(self):
    with self.cached_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=0, dynamic_size=False, infer_shape=True)
      self.assertEqual(0, ta.size().eval())
      # Don't actually perform the pack.  This stores the static shape.
      if control_flow_util.ENABLE_CONTROL_FLOW_V2:
        ta = ta.unstack(array_ops.zeros([0, 3, 5]))
      else:
        ta.unstack(array_ops.zeros([0, 3, 5])).mark_used()
      packed = ta.stack()
      concatenated = ta.concat()
      self.assertAllEqual([0, 3, 5], self.evaluate(packed).shape)
      # Concatenating zero tensors along their first dimension gives a
      # first dimension of zero
      self.assertAllEqual([0, 5], self.evaluate(concatenated).shape)

  @test_util.run_deprecated_v1
  def testSkipEagerTensorArrayEvalEmptyWithDefault(self):
    self._testTensorArrayEvalEmptyWithDefault()

  @test_util.run_deprecated_v1
  def testSkipEagerTensorArrayScatterReadAndGradients(self):
    with self.session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True)

      indices = constant_op.constant([1, 8])
      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.scatter(indices, value)
      r0 = w.read(1)
      r1 = w.read(8)

      # Test combined gradients + aggregation of read(0)
      grad = gradients_impl.gradients(
          ys=[r0, r1], xs=[value], grad_ys=[[2.0, 3.0], [4.0, 5.0]])
      read_vals, grad_vals = session.run([[r0, r1], grad])

      self.assertEqual(len(read_vals), 2)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([1.0, -1.0], read_vals[0])
      self.assertAllEqual([10.0, -10.0], read_vals[1])
      self.assertAllEqual([[2.0, 3.0], [4.0, 5.0]], grad_vals[0])

  @test_util.run_deprecated_v1
  def testSkipEagerTensorArrayScatterPartialReadAndGradients(self):
    with self.session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True)

      indices = constant_op.constant([1, 8])
      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.scatter(indices, value)
      r0 = w.read(1)

      # Test combined gradients + aggregation of read(0)
      grad = gradients_impl.gradients(
          ys=[r0], xs=[value], grad_ys=[[2.0, 3.0]])[0]
      read_val, grad_val = session.run([r0, grad])

      self.assertAllEqual([1.0, -1.0], read_val)
      self.assertAllEqual([[2.0, 3.0], [0.0, 0.0]], grad_val)

  def testScatterIntoExistingList(self):
    ta = tensor_array_ops.TensorArray(
        dtype=dtypes.float32, tensor_array_name="foo", size=5)

    ta = ta.scatter(indices=[3, 4], value=array_ops.ones([2]))
    self.assertAllEqual(ta.stack(), [0., 0., 0., 1., 1.])

    ta = ta.scatter(indices=[1], value=array_ops.ones([1]))
    self.assertAllEqual(ta.stack(), [0., 1., 0., 1., 1.])

    ta = ta.scatter(indices=[0, 2], value=[5., 6.])
    self.assertAllEqual(ta.stack(), [5., 1., 6., 1., 1.])

  @test_util.run_v1_only("b/118890905")
  def testTensorArrayWriteGatherAndGradients(self):
    with self.session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True)

      def func(values):
        indices = constant_op.constant([1, 8])
        w = ta.unstack(values)
        g = w.gather(indices)
        return g

      values = constant_op.constant([[1.0 * x, -1.0 * x] for x in range(10)])
      g = func(values)
      grad_ys = [[[2.0, 3.0], [4.0, 5.0]]]
      # Test combined gradients + aggregation of read(0)
      if context.executing_eagerly():
        g_vals = [g]
        grad_vals = backprop.gradients_function(func)(
            values, dy=constant_op.constant(grad_ys[0], dtype=dtypes.float32))
      else:
        grad = gradients_impl.gradients(ys=[g], xs=[values], grad_ys=grad_ys)
        g_vals, grad_vals = session.run([[g], grad])

      # Gradients for 8 of the 10 unread components are zero.
      expected_grad = np.zeros((10, 2))
      expected_grad[1] = [2.0, 3.0]
      expected_grad[8] = [4.0, 5.0]

      self.assertEqual(len(g_vals), 1)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[1.0, -1.0], [8.0, -8.0]], g_vals[0])
      self.assertAllEqual(expected_grad, grad_vals[0])

  @test_util.disable_control_flow_v2("colocate_with not supported in v2.")
  @test_util.run_v1_only("b/120545219")
  def testSkipEagerTensorArrayGetsDeviceFromFirstWrite(self):
    with ops.device("/job:worker/task:0/cpu:0"):
      # this initial device will be ignored.
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
    with ops.device("/job:worker/task:1/cpu:0"):
      # the first write sets the op's device.
      ta = ta.write(0, 1.0)
    with ops.device("/job:worker/task:2/cpu:0"):
      # subsequent writes do not modify the op's device.
      ta = ta.write(1, 1.0)

    # The gradient TA will sit on the same device as the forward TA.
    ta_grad = ta.grad("grad")
    flows = [ta.flow, ta_grad.flow]

    # Similar tests for unpack and split
    with ops.device("/job:worker/task:0/cpu:0"):
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=3)
    with ops.device("/job:worker/task:1/cpu:0"):
      ta = ta.unstack([1.0, 2.0])
    with ops.device("/job:worker/task:2/cpu:0"):
      ta = ta.write(2, 3.0)
    flows.append(ta.flow)

    with ops.device("/job:worker/task:0/cpu:0"):
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
    with ops.device("/job:worker/task:1/cpu:0"):
      ta = ta.split([1.0, 2.0], [1, 1])
    flows.append(ta.flow)

    session = session_lib.Session(self._workers[0].target)

    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    session.run(flows, options=run_options, run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField("step_stats"))
    dev_stats = {d.device: d.node_stats
                 for d in run_metadata.step_stats.dev_stats}
    for d in dev_stats:
      if "/task:1/" in d:
        self.assertTrue(
            [s for s in dev_stats[d] if "/TensorArray" in s.node_name])
      elif "/host:CPU" not in d:
        self.assertFalse(
            [s for s in dev_stats[d] if "/TensorArray" in s.node_name])

  @test_util.disable_control_flow_v2("colocate_with not supported in v2.")
  @test_util.run_v1_only("b/120545219")
  def testSkipEagerTensorArrayGetsDeviceFromFirstWriteInWhileLoop(self):
    with ops.device("/job:worker/task:0/cpu:0"):
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)

    def _body(i, ta_i):
      with ops.device("/job:worker/task:1/cpu:0"):
        return i + 1, ta_i.write(i, constant_op.constant(0.0))

    _, ta_out = control_flow_ops.while_loop(
        lambda i, ta: i < 2, _body, loop_vars=[0, ta])

    session = session_lib.Session(self._workers[0].target)

    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    session.run(ta_out.flow, options=run_options, run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField("step_stats"))
    dev_stats = {d.device: d.node_stats
                 for d in run_metadata.step_stats.dev_stats}
    for d in dev_stats:
      if "/task:1/" in d:
        self.assertTrue(
            [s for s in dev_stats[d] if "TensorArray" == s.node_name])
      else:
        self.assertFalse(
            [s for s in dev_stats[d] if "TensorArray" == s.node_name])

  @test_util.disable_control_flow_v2("colocate_with not supported in v2.")
  @test_util.run_v1_only("b/120545219")
  def testSkipEagerTensorArrayDisabledColocateWithFirstWriteCall(self):
    with ops.device("/job:worker/task:0/cpu:0"):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=2, colocate_with_first_write_call=False)

    def _body(i, ta_i):
      with ops.device("/job:worker/task:1/cpu:0"):
        return i + 1, ta_i.write(i, constant_op.constant(0.0))

    _, ta_out = control_flow_ops.while_loop(
        lambda i, ta: i < 2, _body, loop_vars=[0, ta])

    session = session_lib.Session(self._workers[0].target)

    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    session.run(ta_out.flow, options=run_options, run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField("step_stats"))
    dev_stats = {d.device: list(d.node_stats)
                 for d in run_metadata.step_stats.dev_stats}
    for d in dev_stats:
      if "/task:0/" in d and "CPU" in d:  # Skip any GPU node stats
        self.assertTrue(
            [s for s in dev_stats[d] if "TensorArray" == s.node_name])
      else:
        self.assertFalse(
            [s for s in dev_stats[d] if "TensorArray" == s.node_name])

  def testTensorArrayIdentity(self):
    with self.session(use_gpu=True):
      ta0 = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2,
                                         infer_shape=False)
      ta1 = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=4,
                                         infer_shape=True)

      ta0 = ta0.write(0, 0.)
      ta1 = ta1.write(0, 1)

      v0 = variable_scope.get_variable(
          "v0", shape=(), initializer=init_ops.zeros_initializer())
      v1 = variable_scope.get_variable(
          "v1", shape=(), initializer=init_ops.zeros_initializer())

      with ops.control_dependencies([v0.assign_add(1)]):
        ta0 = ta0.identity()

      with ops.control_dependencies([v1.assign_add(1)]):
        ta1 = ta1.identity()

      read0 = ta0.read(0)
      read1 = ta1.read(0)

      size0 = ta0.size()
      size1 = ta1.size()

      # Tests correct properties on new TensorArrays.
      self.assertEqual(dtypes.float32, ta0.dtype)
      self.assertEqual(dtypes.int32, ta1.dtype)
      if context.executing_eagerly():
        self.assertEqual(tensor_shape.TensorShape([]), read0.get_shape())
      else:
        self.assertEqual(tensor_shape.unknown_shape(), read0.get_shape())
      self.assertEqual(tensor_shape.TensorShape([]), read1.get_shape())

      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())

      read0_v, read1_v, size0_v, size1_v = self.evaluate((read0, read1, size0,
                                                          size1))

      # Tests that the control dependencies was added and executed.
      self.assertEqual(1, self.evaluate(v0))
      self.assertEqual(1, self.evaluate(v1))

      # Tests correct TensorArray.
      self.assertEqual(read0_v, 0)
      self.assertEqual(read1_v, 1)
      self.assertEqual(size0_v, 2)
      self.assertEqual(size1_v, 4)

  @test_util.deprecated_graph_mode_only
  def testSkipEagerTensorArrayGradYsInCorrectScope(self):
    n_time = 1
    n_dim = 1
    x = constant_op.constant([[1.42]])
    dy = constant_op.constant([[2.42]])

    ta = tensor_array_ops.TensorArray(
        dtypes.float32, size=n_time, element_shape=[n_dim])
    for t in range(n_time):
      ta = ta.write(index=t, value=x[t])
      y = ta.stack()
      # dy is outside of the gradients name scope; tf.gradients must
      # wrap it in the correct name scope.
      dx, = gradients_impl.gradients(ys=[y], xs=[x], grad_ys=[dy])
      with self.cached_session(use_gpu=True) as sess:
        vdx, vdy = self.evaluate([dx, dy])
      self.assertAllClose(vdx, vdy)

  @test_util.deprecated_graph_mode_only
  def testSkipEagerTensorArrayInt64GPU(self):
    if not test.is_gpu_available():
      return
    with self.session(use_gpu=True, force_gpu=True) as sess:
      value = array_ops.placeholder(dtypes.int64)
      ta = tensor_array_ops.TensorArray(dtype=dtypes.int64, size=2)
      ta = ta.scatter([0, 1], value)
      r0 = ta.read(0)
      r1 = ta.read(1)
      v0, v1 = sess.run([r0, r1], feed_dict={value: [-3, 100]})
      self.assertAllEqual(v0, -3)
      self.assertAllEqual(v1, 100)

  def testInferShapeFalseValid(self):
    ta = tensor_array_ops.TensorArray(
        dtypes.float32, size=3, infer_shape=False, element_shape=[None, 10, 20])
    ta = ta.write(0, array_ops.ones([50, 10, 20]))
    ta = ta.write(1, array_ops.ones([50, 10, 20]))
    ta = ta.write(2, array_ops.ones([1, 10, 20]))
    ta = ta.concat()

    correct = np.ones([101, 10, 20])

    self.assertAllEqual(ta, correct)

  def testInferShapeFalseInvalid(self):
    ta = tensor_array_ops.TensorArray(
        dtypes.float32, size=2, infer_shape=False, element_shape=[None, 10, 20])
    ta = ta.write(0, array_ops.ones([50, 10, 20]))

    with self.assertRaises(ValueError):
      ta = ta.write(1, array_ops.ones([1, 20, 20]))

  def testInferShapeTrue(self):
    ta = tensor_array_ops.TensorArray(
        dtypes.float32, size=3, infer_shape=True, element_shape=[None, 10, 20])
    self.assertAllEqual((None, 10, 20), ta.element_shape.as_list())
    ta = ta.write(0, array_ops.ones([50, 10, 20]))
    self.assertAllEqual((50, 10, 20), ta.element_shape.as_list())
    ta = ta.write(1, array_ops.ones([50, 10, 20]))
    with self.assertRaises(ValueError):
      ta = ta.write(
          2, array_ops.ones([1, 10, 20])
      )  # Inconsistent shapes: saw (1, 10, 20) but expected (50, 10, 20)

  def testStackShapeOnEmpty(self):
    ta = tensor_array_ops.TensorArray(
        dtypes.float32, size=0, element_shape=(5, 10), dynamic_size=True)
    self.assertAllEqual([0, 5, 10], self.evaluate(ta.stack()).shape)

  @test_util.run_deprecated_v1
  def testSkipEagerStackOnPartiallyDefinedShape(self):
    ta = tensor_array_ops.TensorArray(
        dtypes.float32, size=0, element_shape=(5, None), dynamic_size=True)
    self.assertEqual([None, 5, None], ta.stack().shape.as_list())

  def testStackShapeOnStaticSize(self):
    ta = tensor_array_ops.TensorArray(dtypes.float32, size=42)
    ta = ta.write(0, [0])
    self.assertEqual([42, 1], ta.stack().shape.as_list())


class TensorArrayBenchmark(test.Benchmark):

  def _tensorArrayWriteInWhile(self):
    size = 10000
    ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=size)
    (_, ta) = control_flow_ops.while_loop(
        lambda i, _: i < size,
        lambda i, ta: (i + 1, ta.write(i, 0.)), [0, ta],
        parallel_iterations=1)
    return ta.stack()

  def _benchmarkWriteInWhile(self):
    ops.reset_default_graph()
    op = self._tensorArrayWriteInWhile()
    self.run_op_benchmark(session_lib.Session(), op)

  def benchmarkWriteInWhile(self):
    self._benchmarkWriteInWhile()

  @test_util.enable_control_flow_v2
  def benchmarkWriteInWhileWithControlFlowV2(self):
    self._benchmarkWriteInWhile()

  def benchmarkWriteInDatasetMapFn(self):
    ds = dataset_ops.Dataset.from_tensors(array_ops.zeros([10])).repeat()
    ds = ds.map(lambda _: self._tensorArrayWriteInWhile())
    op = ds.make_one_shot_iterator().get_next()
    self.run_op_benchmark(session_lib.Session(), op)

  def benchmarkWriteInDatasetParallelMapFn(self):
    ds = dataset_ops.Dataset.from_tensors(array_ops.zeros([10])).repeat()
    ds = ds.map(lambda _: self._tensorArrayWriteInWhile(), num_parallel_calls=2)
    op = ds.make_one_shot_iterator().get_next()
    self.run_op_benchmark(session_lib.Session(), op)


if __name__ == "__main__":
  test.main()
