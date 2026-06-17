# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sharding util ops (XlaSplitND, XlaConcatND)."""

from typing import Any, List, Optional

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.client.session import Session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.framework.tensor import Tensor
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def create_tensor_split_graph(
    sess: Session,
    input_value: Any,
    input_dtype: Any,
    num_outputs: int,
    num_splits: List[int],
    paddings: Optional[List[int]] = None) -> List[Tensor]:
  del sess

  const_input_op = constant_op.constant(input_value, dtype=input_dtype)
  return gen_tpu_ops.xla_split_nd(
      const_input_op, num_outputs, num_splits, paddings=paddings)


def create_resource_split_graph(
    sess: Session,
    input_value: Any,
    input_dtype: Any,
    num_outputs: int,
    num_splits: List[int],
    paddings: Optional[List[int]] = None) -> List[Tensor]:
  variable = resource_variable_ops.ResourceVariable(
      initial_value=input_value, dtype=input_dtype)
  sess.run(variables.variables_initializer([variable]))
  return gen_tpu_ops.read_variable_xla_split_nd(
      variable.handle, input_dtype, num_outputs, num_splits, paddings=paddings)


class XlaSplitNDOpTest(xla_test.XLATestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitDimensionZero(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[[0]]],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[1, 1, 0])
        with self.assertRaisesOpError('index 2 must be positive, but got 0'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitDimensionNegative(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[[0]]],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[1, -1, 1])
        with self.assertRaisesOpError('index 1 must be positive, but got -1'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testNumOutputsMismatch(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[0, 1],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[2])
        with self.assertRaisesOpError('\'N\' must match number of slices 2'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testPaddingsLengthMismatch(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3]],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[0])
        with self.assertRaisesOpError('length 2, but got 1'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testPaddingsNegative(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3]],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[0, -1])
        with self.assertRaisesOpError('non-negative, but got -1 at index 1'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testInputRankSplitMismatch(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3]],
            input_dtype=dtype,
            num_outputs=8,
            num_splits=[2, 2, 2])
        with self.assertRaisesOpError(
            '\'num_splits\' length 3, but got rank 2'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testDimNotEvenlySplit(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3], [4, 5], [6, 7]],
            input_dtype=dtype,
            num_outputs=6,
            num_splits=[3, 2])
        with self.assertRaisesOpError('divisible by \'num_splits\' 3'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testDimWithPaddingNotEvenlySplit(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0, 1], [2, 3], [4, 5], [6, 7]],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[0, 1])
        with self.assertRaisesOpError('divisible by \'num_splits\' 2'):
          sess.run(split)

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testNoSplits(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[1, 1, 1])
        results = sess.run(split)
      self.assertLen(results, 1)
      self.assertAllClose(results[0], [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testNoSplitsWithPadding(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[[0]], [[1]]],
            input_dtype=dtype,
            num_outputs=1,
            num_splits=[1, 1, 1],
            paddings=[0, 1, 1])
        results = sess.run(split)
      self.assertLen(results, 1)
      self.assertAllClose(results[0], [[[0, 0], [0, 0]], [[1, 0], [0, 0]]])

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitNoPadding(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2])
        results = sess.run(split)
      self.assertLen(results, 4)
      self.assertAllClose(results[0], [[0, 1], [4, 5]])
      self.assertAllClose(results[1], [[2, 3], [6, 7]])
      self.assertAllClose(results[2], [[8, 9], [12, 13]])
      self.assertAllClose(results[3], [[10, 11], [14, 15]])

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitPartialPadding(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[1, 1])
        results = sess.run(split)
      self.assertLen(results, 4)
      self.assertAllClose(results[0], [[0, 1], [3, 4]])
      self.assertAllClose(results[1], [[2, 0], [5, 0]])
      self.assertAllClose(results[2], [[6, 7], [0, 0]])
      self.assertAllClose(results[3], [[8, 0], [0, 0]])

  @parameterized.named_parameters(('Tensor', create_tensor_split_graph),
                                  ('Resource', create_resource_split_graph))
  def testSplitCompletePadding(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=[[0], [1]],
            input_dtype=dtype,
            num_outputs=4,
            num_splits=[2, 2],
            paddings=[2, 3])
        results = sess.run(split)
      self.assertLen(results, 4)
      self.assertAllClose(results[0], [[0, 0], [1, 0]])
      self.assertAllClose(results[1], [[0, 0], [0, 0]])
      self.assertAllClose(results[2], [[0, 0], [0, 0]])
      self.assertAllClose(results[3], [[0, 0], [0, 0]])

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_split_graph, 1),
      ('2Tensor', create_tensor_split_graph, 2),
      ('3Tensor', create_tensor_split_graph, 3),
      ('4Tensor', create_tensor_split_graph, 4),
      ('5Tensor', create_tensor_split_graph, 5),
      ('6Tensor', create_tensor_split_graph, 6),
      ('7Tensor', create_tensor_split_graph, 7),
      ('8Tensor', create_tensor_split_graph, 8),
      ('1Resource', create_resource_split_graph, 1),
      ('2Resource', create_resource_split_graph, 2),
      ('3Resource', create_resource_split_graph, 3),
      ('4Resource', create_resource_split_graph, 4),
      ('5Resource', create_resource_split_graph, 5),
      ('6Resource', create_resource_split_graph, 6),
      ('7Resource', create_resource_split_graph, 7),
      ('8Resource', create_resource_split_graph, 8),
  )
  def testRanked(self, graph_fn, rank):
    num_splits = [2] * rank
    num_outputs = 2 << (rank - 1)
    input_value = np.reshape(np.arange(np.prod(num_splits)), num_splits)
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        split = graph_fn(
            sess,
            input_value=input_value,
            input_dtype=dtype,
            num_outputs=num_outputs,
            num_splits=num_splits)
        results = sess.run(split)
      self.assertLen(results, num_outputs)
      for i, result in enumerate(results):
        expected_output = np.reshape(i, [1] * rank).astype(dtype)
        self.assertAllClose(result, expected_output)


def create_tensor_concat_graph(
    sess: Session,
    input_values: List[Any],
    input_dtype: Any,
    num_concats: List[int],
    paddings: Optional[List[int]] = None,
    output_shape: Optional[List[int]] = None) -> Tensor:
  del sess
  del output_shape

  const_input_ops = [
      constant_op.constant(i, dtype=input_dtype) for i in input_values
  ]
  return gen_tpu_ops.xla_concat_nd(const_input_ops, num_concats, paddings)


def create_resource_concat_graph(
    sess: Session,
    input_values: List[Any],
    input_dtype: Any,
    num_concats: List[int],
    paddings: Optional[List[int]] = None,
    output_shape: Optional[List[int]] = None) -> Tensor:
  variable_shape = [] if output_shape is None else output_shape
  variable = resource_variable_ops.ResourceVariable(
      initial_value=np.zeros(variable_shape, dtype=input_dtype),
      dtype=input_dtype)
  sess.run(variables.variables_initializer([variable]))
  const_input_ops = [
      constant_op.constant(i, dtype=input_dtype) for i in input_values
  ]
  concat = gen_tpu_ops.assign_variable_xla_concat_nd(variable.handle,
                                                     const_input_ops,
                                                     num_concats, paddings)
  with control_dependencies([concat]):
    return variable.read_value()


class XlaConcatNDOpTest(xla_test.XLATestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatDimensionZero(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[[0]]]],
            input_dtype=dtype,
            num_concats=[1, 1, 0])
        with self.assertRaisesOpError('index 2 must be positive, but got 0'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatDimensionNegative(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[[0]]]],
            input_dtype=dtype,
            num_concats=[1, -1, 1])
        with self.assertRaisesOpError('index 1 must be positive, but got -1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testNumInputsMismatch(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess, input_values=[[0, 1]], input_dtype=dtype, num_concats=[2])
        with self.assertRaisesOpError('\'N\' must match number of slices 2'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testPaddingsLengthMismatch(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[0, 1], [2, 3]]],
            input_dtype=dtype,
            num_concats=[1, 1],
            paddings=[0])
        with self.assertRaisesOpError('length 2, but got 1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testPaddingsNegative(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[0, 1], [2, 3]]],
            input_dtype=dtype,
            num_concats=[1, 1],
            paddings=[0, -1])
        with self.assertRaisesOpError('non-negative, but got -1 at index 1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testInputRankConcatMismatch(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess, input_values=[[0]], input_dtype=dtype, num_concats=[1, 1])
        with self.assertRaisesOpError(
            '\'num_concats\' length 2, but got rank 1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testDifferentShapedInputs(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[0], [1, 2]],
            input_dtype=dtype,
            num_concats=[2])
        with self.assertRaisesOpError(
            r'same expected shape \[1\], but got \[2\] at index 1'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testPaddingExceedsOutputDimSize(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[0]],
            input_dtype=dtype,
            num_concats=[1],
            paddings=[2])
        with self.assertRaisesOpError(
            'exceed expected output shape dimension 1 at index 0, but got 2'):
          sess.run(concat)

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testNoConcats(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]],
            input_dtype=dtype,
            num_concats=[1, 1, 1],
            output_shape=[2, 2, 2])
        result = sess.run(concat)
      self.assertAllClose(result, [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testNoConcatsWithPadding(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]],
            input_dtype=dtype,
            num_concats=[1, 1, 1],
            output_shape=[1, 1, 1],
            paddings=[1, 1, 1])
        result = sess.run(concat)
      self.assertAllClose(result, [[[0]]])

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatNoPadding(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            input_dtype=dtype,
            num_concats=[2, 2],
            output_shape=[4, 4])
        result = sess.run(concat)
      self.assertAllClose(
          result,
          [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatPartialPadding(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            input_dtype=dtype,
            num_concats=[2, 2],
            output_shape=[3, 3],
            paddings=[1, 1])
        result = sess.run(concat)
      self.assertAllClose(result, [[0, 1, 4], [2, 3, 6], [8, 9, 12]])

  @parameterized.named_parameters(('Tensor', create_tensor_concat_graph),
                                  ('Resource', create_resource_concat_graph))
  def testConcatCompletePadding(self, graph_fn):
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=[
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[12, 13], [14, 15]],
            ],
            input_dtype=dtype,
            num_concats=[2, 2],
            output_shape=[2, 2],
            paddings=[2, 2])
        result = sess.run(concat)
      self.assertAllClose(result, [[0, 1], [2, 3]])

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_concat_graph, 1),
      ('2Tensor', create_tensor_concat_graph, 2),
      ('3Tensor', create_tensor_concat_graph, 3),
      ('4Tensor', create_tensor_concat_graph, 4),
      ('5Tensor', create_tensor_concat_graph, 5),
      ('6Tensor', create_tensor_concat_graph, 6),
      ('7Tensor', create_tensor_concat_graph, 7),
      ('8Tensor', create_tensor_concat_graph, 8),
      ('1Resource', create_resource_concat_graph, 1),
      ('2Resource', create_resource_concat_graph, 2),
      ('3Resource', create_resource_concat_graph, 3),
      ('4Resource', create_resource_concat_graph, 4),
      ('5Resource', create_resource_concat_graph, 5),
      ('6Resource', create_resource_concat_graph, 6),
      ('7Resource', create_resource_concat_graph, 7),
      ('8Resource', create_resource_concat_graph, 8),
  )
  def testRanked(self, graph_fn, rank):
    num_concats = [2] * rank
    num_inputs = 2 << (rank - 1)
    input_values = [np.reshape(i, [1] * rank) for i in range(num_inputs)]
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        concat = graph_fn(
            sess,
            input_values=input_values,
            input_dtype=dtype,
            num_concats=num_concats,
            output_shape=num_concats)
        result = sess.run(concat)
      expected_output = np.arange(0,
                                  num_inputs).reshape(num_concats).astype(dtype)
      self.assertAllClose(result, expected_output)


def create_tensor_roundtrip_graph(
    sess: Session,
    value: Any,
    dtype: Any,
    num_partitions: List[int],
    paddings: Optional[List[int]] = None) -> Tensor:
  del sess

  const_input_op = constant_op.constant(value, dtype=dtype)
  split = gen_tpu_ops.xla_split_nd(
      const_input_op,
      np.prod(num_partitions),
      num_partitions,
      paddings=paddings)
  concat = gen_tpu_ops.xla_concat_nd(split, num_partitions, paddings)
  return math_ops.equal(const_input_op, concat)


def create_resource_roundtrip_graph(
    sess: Session,
    value: Any,
    dtype: Any,
    num_partitions: List[int],
    paddings: Optional[List[int]] = None) -> Tensor:
  variable = resource_variable_ops.ResourceVariable(
      initial_value=value, dtype=dtype)
  sess.run(variables.variables_initializer([variable]))
  split = gen_tpu_ops.read_variable_xla_split_nd(
      variable.handle,
      dtype,
      np.prod(num_partitions),
      num_partitions,
      paddings=paddings)
  concat = gen_tpu_ops.assign_variable_xla_concat_nd(variable.handle, split,
                                                     num_partitions, paddings)
  with control_dependencies([concat]):
    return math_ops.equal(variable.read_value(),
                          constant_op.constant(value, dtype=dtype))


class XlaSplitConcatNDTest(xla_test.XLATestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_roundtrip_graph, 1),
      ('2Tensor', create_tensor_roundtrip_graph, 2),
      ('3Tensor', create_tensor_roundtrip_graph, 3),
      ('4Tensor', create_tensor_roundtrip_graph, 4),
      ('5Tensor', create_tensor_roundtrip_graph, 5),
      ('6Tensor', create_tensor_roundtrip_graph, 6),
      ('7Tensor', create_tensor_roundtrip_graph, 7),
      ('8Tensor', create_tensor_roundtrip_graph, 8),
      ('1Resource', create_resource_roundtrip_graph, 1),
      ('2Resource', create_resource_roundtrip_graph, 2),
      ('3Resource', create_resource_roundtrip_graph, 3),
      ('4Resource', create_resource_roundtrip_graph, 4),
      ('5Resource', create_resource_roundtrip_graph, 5),
      ('6Resource', create_resource_roundtrip_graph, 6),
      ('7Resource', create_resource_roundtrip_graph, 7),
      ('8Resource', create_resource_roundtrip_graph, 8),
  )
  def testNoPadding(self, graph_fn, rank):
    num_partitions = [2] * rank
    shape = [4] * rank
    value = np.arange(0, np.prod(shape)).reshape(shape)
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        validate = graph_fn(sess, value, dtype, num_partitions)
        result = sess.run(validate)
      self.assertAllEqual(result, np.broadcast_to(True, shape))

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_roundtrip_graph, 1),
      ('2Tensor', create_tensor_roundtrip_graph, 2),
      ('3Tensor', create_tensor_roundtrip_graph, 3),
      ('4Tensor', create_tensor_roundtrip_graph, 4),
      ('5Tensor', create_tensor_roundtrip_graph, 5),
      ('6Tensor', create_tensor_roundtrip_graph, 6),
      ('7Tensor', create_tensor_roundtrip_graph, 7),
      ('8Tensor', create_tensor_roundtrip_graph, 8),
      ('1Resource', create_resource_roundtrip_graph, 1),
      ('2Resource', create_resource_roundtrip_graph, 2),
      ('3Resource', create_resource_roundtrip_graph, 3),
      ('4Resource', create_resource_roundtrip_graph, 4),
      ('5Resource', create_resource_roundtrip_graph, 5),
      ('6Resource', create_resource_roundtrip_graph, 6),
      ('7Resource', create_resource_roundtrip_graph, 7),
      ('8Resource', create_resource_roundtrip_graph, 8),
  )
  def testPartialPadding(self, graph_fn, rank):
    num_partitions = [2] * rank
    shape = [4] * rank
    value = np.arange(0, np.prod(shape)).reshape(shape)
    paddings = [2] * rank
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        validate = graph_fn(sess, value, dtype, num_partitions, paddings)
        result = sess.run(validate)
      self.assertAllEqual(result, np.broadcast_to(True, shape))

  @parameterized.named_parameters(
      ('1Tensor', create_tensor_roundtrip_graph, 1),
      ('2Tensor', create_tensor_roundtrip_graph, 2),
      ('3Tensor', create_tensor_roundtrip_graph, 3),
      ('4Tensor', create_tensor_roundtrip_graph, 4),
      ('5Tensor', create_tensor_roundtrip_graph, 5),
      ('6Tensor', create_tensor_roundtrip_graph, 6),
      ('7Tensor', create_tensor_roundtrip_graph, 7),
      ('8Tensor', create_tensor_roundtrip_graph, 8),
      ('1Resource', create_resource_roundtrip_graph, 1),
      ('2Resource', create_resource_roundtrip_graph, 2),
      ('3Resource', create_resource_roundtrip_graph, 3),
      ('4Resource', create_resource_roundtrip_graph, 4),
      ('5Resource', create_resource_roundtrip_graph, 5),
      ('6Resource', create_resource_roundtrip_graph, 6),
      ('7Resource', create_resource_roundtrip_graph, 7),
      ('8Resource', create_resource_roundtrip_graph, 8),
  )
  def testCompletePadding(self, graph_fn, rank):
    num_partitions = [2] * rank
    shape = [4] * rank
    value = np.arange(0, np.prod(shape)).reshape(shape)
    paddings = [4] * rank
    for dtype in self.numeric_types:
      with self.session() as sess, self.device_scope():
        validate = graph_fn(sess, value, dtype, num_partitions, paddings)
        result = sess.run(validate)
      self.assertAllEqual(result, np.broadcast_to(True, shape))


if __name__ == '__main__':
  test.main()
