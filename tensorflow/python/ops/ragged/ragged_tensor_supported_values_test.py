# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for RaggedTensor supported value types."""

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_test_ops as test_ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.platform import googletest
from tensorflow.python.util import dispatch


class WrappedTensor(extension_type.BatchableExtensionType):
  value: ops.Tensor

  @property
  def shape(self):
    return self.value.shape

  @property
  def dtype(self):
    return self.value.dtype

  def __getitem__(self, idx):
    return WrappedTensor(self.value.__getitem__(idx))

  def set_shape(self, shape):
    return self.value.set_shape(shape)

  class Spec(type_spec.TypeSpec):

    @property
    def shape(self):
      return self.value.shape

    @property
    def dtype(self):
      return self.value.dtype


class WrappedTensorOpDispatcher(dispatch.GlobalOpDispatcher):
  """Global op dispatcher for WrappedTensor."""

  # For these ops, just return plain Tensors (not WrappedTensors).
  OPS_THAT_RETURN_UNTRACED_RESULTS = (array_ops.shape, array_ops.shape_v2,
                                      check_ops.assert_rank_at_least)

  def call_op(self, op, *args, **kwargs):
    return op(*args, **kwargs)

  def handle(self, op, args, kwargs):
    # Dispatcher only applies if at least one arg is a WrappedTensor.
    if not (any(self.is_wrapped_tensor_arg(x) for x in args) or
            any(self.is_wrapped_tensor_arg(x) for x in kwargs.values())):
      return self.NOT_SUPPORTED

    args = [self.unwrap(v) for v in args]
    kwargs = dict([(k, self.unwrap(v)) for (k, v) in kwargs.items()])
    value = self.call_op(op, *args, **kwargs)
    if op in self.OPS_THAT_RETURN_UNTRACED_RESULTS:
      return value
    else:
      return WrappedTensor(value)

  def unwrap(self, value):
    if isinstance(value, WrappedTensor):
      return value.value
    elif isinstance(value, (list, tuple)):
      return type(value)([self.unwrap(v) for v in value])
    else:
      return value

  def is_wrapped_tensor_arg(self, value):
    if isinstance(value, WrappedTensor):
      return True
    if isinstance(value, (list, tuple)):
      if any(isinstance(x, WrappedTensor) for x in value):
        return True
    return False


WrappedTensorOpDispatcher().register()
ragged_tensor._add_supported_value_type(WrappedTensor)


# pylint: disable=g-complex-comprehension
@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorSupportedValuesTest(test_util.TensorFlowTestCase,
                                      parameterized.TestCase):

  def assertAllTensorsEqual(self, list1, list2):
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    tensor_values = constant_op.constant(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    values = WrappedTensor(tensor_values)

    row_splits = constant_op.constant([0, 2, 2, 5, 6, 8], dtypes.int64)
    rt = RaggedTensor.from_row_splits(values, row_splits)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.row_splits, row_splits)

    row_starts = constant_op.constant([0, 2, 2, 5, 6], dtypes.int64)
    rt = RaggedTensor.from_row_starts(values, row_starts)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.row_starts(), row_starts)

    row_limits = constant_op.constant([2, 2, 5, 6, 8], dtypes.int64)
    rt = RaggedTensor.from_row_limits(values, row_limits)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.row_limits(), row_limits)

    row_lengths = constant_op.constant([2, 0, 3, 1, 2], dtypes.int64)
    rt = RaggedTensor.from_row_lengths(values, row_lengths)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.row_lengths(), row_lengths)

    rt = RaggedTensor.from_uniform_row_length(values, 4)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.uniform_row_length, 4)

  def testWithValues(self):
    tensor_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    values = WrappedTensor(tensor_values)
    nested_row_splits = [[0, 2, 5], [0, 2, 2, 5, 6, 7]]
    rt = RaggedTensor.from_nested_row_splits(values, nested_row_splits)

    tensor_int = constant_op.constant([1, 2, 3, 4, 5])
    rt_int = rt.with_values(tensor_int)
    self.assertAllEqual(rt_int.values, tensor_int)

    rt_wrapped_int = rt.with_values(WrappedTensor(tensor_int))
    self.assertIsInstance(rt_wrapped_int.values, WrappedTensor)
    self.assertAllEqual(rt_wrapped_int.values.value, tensor_int)

  def testWithFlatValues(self):
    tensor_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    values = WrappedTensor(tensor_values)
    nested_row_splits = [[0, 2, 5], [0, 2, 2, 5, 6, 7]]
    rt = RaggedTensor.from_nested_row_splits(values, nested_row_splits)

    tensor_int = constant_op.constant([1, 2, 3, 4, 5, 6, 7])
    rt_int = rt.with_flat_values(tensor_int)
    self.assertAllEqual(rt_int.flat_values, tensor_int)

    rt_wrapped_int = rt.with_flat_values(WrappedTensor(tensor_int))
    self.assertIsInstance(rt_wrapped_int.flat_values, WrappedTensor)
    self.assertAllEqual(rt_wrapped_int.flat_values.value, tensor_int)

  @parameterized.parameters(
      #=========================================================================
      # Test each unary op.
      #=========================================================================
      [{'x': ([[-2.0, 3.0], [-3.0]]), 'op': op}
       for op in test_ops.UNARY_FLOAT_OPS] +
      [{'x': ([[True, False], [True]]),
        'op': op}
       for op in test_ops.UNARY_BOOL_OPS] +
      [{'x': [[18, 512], [12412]],
        'x_dtype': dtypes.int32,
        'op': op}
       for op in test_ops.UNARY_INT_OPS] +
      [{'x': ([['abcd', 'efgh'], ['aabbccdd']]),
        'op': op}
       for op in test_ops.UNARY_STRING_OPS] +
      [
          {'op': clip_ops.clip_by_value,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'clip_value_min': 0.1, 'clip_value_max': 4.0},
          {'op': math_ops.cast,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'dtype': dtypes.int32},
          {'op': math_ops.saturate_cast,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'dtype': dtypes.int32},
          {'op': string_ops.string_to_hash_bucket,
           'x': (
               [['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000},
          {'op': string_ops.string_to_hash_bucket_fast,
           'x': (
               [['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000},
          {'op': string_ops.string_to_hash_bucket_strong,
           'x': (
               [['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000,
           'key': [1231, 12512]},
          {'op': string_ops.string_to_number,
           'x': ([['-2.0', '3.0'], ['-3.0']])},
          {'op': string_ops.regex_full_match,
           'x': ([['hello', '123'], ['1+1']]),
           'pattern': r'\w+'},
          {'op': string_ops.regex_replace,
           'x': ([['hello', '123'], ['1+1']]),
           'pattern': r'\d',
           'rewrite': '#'},
          {'op': string_ops.substr,
           'x': ([['hello', '123'], ['1+1']]),
           'pos': 2, 'len': 3},
          {'op': array_ops.check_numerics,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'message': 'check-numerics'},
          {'op': nn_ops.dropout,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'rate': 0.5,
           'seed': 1},
          {'op': array_ops.expand_dims_v2,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'axis': -1},
      ])  # pyformat: disable
  def testUnaryElementwiseOp(self,
                             x,
                             x_dtype=None,
                             op=math_ops.abs,
                             **extra_args):
    x = ragged_factory_ops.constant(x, x_dtype)
    wrapped_x = ragged_tensor.RaggedTensor.from_nested_row_splits(
        WrappedTensor(x.flat_values), x.nested_row_splits)
    test_util.random_seed.set_seed(1234)
    res = op(x, **extra_args)
    test_util.random_seed.set_seed(1234)
    wrapped_res = op(wrapped_x, **extra_args)
    self.assertIsInstance(wrapped_res.flat_values, WrappedTensor)
    self.assertAllEqual(wrapped_res.flat_values.value, res.flat_values)
    self.assertAllTensorsEqual(wrapped_res.nested_row_splits,
                               res.nested_row_splits)

  @parameterized.parameters(
      #=========================================================================
      # Test each binary op.
      #=========================================================================
      [{'x': [[-2.0, 3.0], [-3.0]],
        'y': [[5.0, 1.0], [12.0]],
        'op': op}
       for op in test_ops.BINARY_FLOAT_OPS] +
      [{'x': [[-2, 3], [-3]],
        'y': [[5, 1], [12]],
        'op': op}
       for op in test_ops.BINARY_INT_OPS] +
      [{'x': [[True, True], [False]],
        'y': [[False, True], [False]],
        'op': op}
       for op in test_ops.BINARY_BOOL_OPS]
      )  # pyformat: disable
  def testBinaryElementwiseOp(self, x, y, op=math_ops.add):
    x = ragged_factory_ops.constant(x)
    y = ragged_factory_ops.constant(y)
    wrapped_x = ragged_tensor.RaggedTensor.from_nested_row_splits(
        WrappedTensor(x.flat_values), x.nested_row_splits)
    wrapped_y = ragged_tensor.RaggedTensor.from_nested_row_splits(
        WrappedTensor(y.flat_values), y.nested_row_splits)
    res = op(x, y)
    wrapped_res = op(wrapped_x, wrapped_y)
    self.assertIsInstance(wrapped_res.flat_values, WrappedTensor)
    self.assertAllEqual(wrapped_res.flat_values.value, res.flat_values)
    self.assertAllTensorsEqual(wrapped_res.nested_row_splits,
                               res.nested_row_splits)


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorSpecSupportedValuesTest(test_util.TensorFlowTestCase,
                                          parameterized.TestCase):

  def assertAllTensorsEqual(self, list1, list2):
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    flat_values_spec = WrappedTensor.Spec(
        tensor_spec.TensorSpec(shape=(None, 5), dtype=dtypes.float32))
    spec1 = RaggedTensorSpec(
        shape=None,
        dtype=dtypes.float32,
        ragged_rank=1,
        row_splits_dtype=dtypes.int64,
        flat_values_spec=flat_values_spec)
    self.assertIsNone(spec1._shape.rank)
    self.assertEqual(spec1._dtype, dtypes.float32)
    self.assertEqual(spec1._row_splits_dtype, dtypes.int64)
    self.assertEqual(spec1._ragged_rank, 1)
    self.assertEqual(spec1._flat_values_spec, flat_values_spec)

    self.assertIsNone(spec1.shape.rank)
    self.assertEqual(spec1.dtype, dtypes.float32)
    self.assertEqual(spec1.row_splits_dtype, dtypes.int64)
    self.assertEqual(spec1.ragged_rank, 1)
    self.assertEqual(spec1.flat_values_spec, flat_values_spec)

    with self.assertRaisesRegex(
        ValueError, 'dtype must be the same as flat_values_spec.dtype'):
      spec1 = RaggedTensorSpec(
          shape=None,
          dtype=dtypes.float64,
          ragged_rank=1,
          row_splits_dtype=dtypes.int64,
          flat_values_spec=flat_values_spec)

  @parameterized.parameters([
      (RaggedTensorSpec(
          ragged_rank=1,
          flat_values_spec=tensor_spec.TensorSpec(None, dtypes.float32)),
       (tensor_shape.TensorShape(None), dtypes.float32, 1, dtypes.int64,
        tensor_spec.TensorSpec(None, dtypes.float32))),
      (RaggedTensorSpec(
          shape=(5, None, 5),
          ragged_rank=1,
          dtype=dtypes.float64,
          flat_values_spec=tensor_spec.TensorSpec(
              (5,), dtypes.float64)), (tensor_shape.TensorShape(
                  (5, None, 5)), dtypes.float64, 1, dtypes.int64,
                                       tensor_spec.TensorSpec((5,),
                                                              dtypes.float64))),
  ])
  def testSerialize(self, rt_spec, expected):
    serialization = rt_spec._serialize()
    # TensorShape has an unconventional definition of equality, so we can't use
    # assertEqual directly here.  But repr() is deterministic and lossless for
    # the expected values, so we can use that instead.
    self.assertEqual(repr(serialization), repr(expected))

  @parameterized.parameters([
      (RaggedTensorSpec(
          ragged_rank=0,
          shape=[5, 3],
          flat_values_spec=WrappedTensor.Spec(
              tensor_spec.TensorSpec([5, 3], dtypes.float32))),
       [WrappedTensor.Spec(tensor_spec.TensorSpec([5, 3], dtypes.float32))]),
      (RaggedTensorSpec(
          ragged_rank=1,
          flat_values_spec=WrappedTensor.Spec(
              tensor_spec.TensorSpec([None, 3], dtypes.float32))),
       [
           WrappedTensor.Spec(
               tensor_spec.TensorSpec([None, 3], dtypes.float32)),
           tensor_spec.TensorSpec([None], dtypes.int64),
       ]),
      (RaggedTensorSpec(
          ragged_rank=2,
          dtype=dtypes.float64,
          flat_values_spec=WrappedTensor.Spec(
              tensor_spec.TensorSpec([None, 3], dtypes.float64))),
       [
           WrappedTensor.Spec(
               tensor_spec.TensorSpec([None, 3], dtypes.float64)),
           tensor_spec.TensorSpec([None], dtypes.int64),
           tensor_spec.TensorSpec([None], dtypes.int64),
       ]),
      (RaggedTensorSpec(
          shape=[5, None, None],
          dtype=dtypes.string,
          flat_values_spec=WrappedTensor.Spec(
              tensor_spec.TensorSpec([None, 3], dtypes.string))),
       [
           WrappedTensor.Spec(tensor_spec.TensorSpec([None, 3], dtypes.string)),
           tensor_spec.TensorSpec([6], dtypes.int64),
           tensor_spec.TensorSpec([None], dtypes.int64),
       ]),
  ])
  def testComponentSpecs(self, rt_spec, expected):
    self.assertEqual(rt_spec._component_specs, expected)

  @parameterized.parameters([
      {
          'rt_spec':
              RaggedTensorSpec(
                  shape=[3, None, None],
                  ragged_rank=1,
                  flat_values_spec=WrappedTensor.Spec(
                      tensor_spec.TensorSpec(None, dtype=dtypes.float32))),
          'flat_values': [[1.0, 2.0], [3.0, 4.0]],
          'nested_row_splits': [[0, 1, 1, 2]],
      },
      {
          'rt_spec':
              RaggedTensorSpec(
                  shape=[2, None, None],
                  flat_values_spec=WrappedTensor.Spec(
                      tensor_spec.TensorSpec(None, dtype=dtypes.float32))),
          'flat_values': [1.0, 2.0, 3.0, 4.0],
          'nested_row_splits': [[0, 2, 4], [0, 2, 3, 3, 4]],
      },
  ])
  def testToFromComponents(self, rt_spec, flat_values, nested_row_splits):
    wrapped_tensor = WrappedTensor(constant_op.constant(flat_values))
    rt = RaggedTensor.from_nested_row_splits(wrapped_tensor, nested_row_splits)
    components = rt_spec._to_components(rt)
    self.assertIsInstance(components[0], WrappedTensor)
    self.assertAllEqual(components[0].value, wrapped_tensor.value)
    self.assertAllTensorsEqual(components[1:], nested_row_splits)
    rt_reconstructed = rt_spec._from_components(components)
    self.assertIsInstance(rt_reconstructed.flat_values, WrappedTensor)
    self.assertAllEqual(rt_reconstructed.flat_values.value,
                        wrapped_tensor.value)
    self.assertAllTensorsEqual(rt_reconstructed.nested_row_splits,
                               rt.nested_row_splits)
    self.assertEqual(rt_reconstructed.dtype, rt.dtype)

  def testIsCompatibleWith(self):
    spec1 = RaggedTensorSpec([32, None, None],
                             dtypes.float32,
                             2,
                             flat_values_spec=WrappedTensor.Spec(
                                 tensor_spec.TensorSpec([None, None],
                                                        dtypes.float32)))
    spec2 = RaggedTensorSpec(
        None,
        dtypes.float32,
        2,
        flat_values_spec=WrappedTensor.Spec(
            tensor_spec.TensorSpec(None, dtypes.float32)))
    spec3 = RaggedTensorSpec(
        None,
        dtypes.int32,
        1,
        flat_values_spec=WrappedTensor.Spec(
            tensor_spec.TensorSpec(None, dtypes.int32)))
    spec4 = RaggedTensorSpec([None],
                             dtypes.int32,
                             0,
                             flat_values_spec=WrappedTensor.Spec(
                                 tensor_spec.TensorSpec(None, dtypes.int32)))
    spec5 = RaggedTensorSpec([None], dtypes.int32, 0)

    self.assertTrue(spec1.is_compatible_with(spec2))
    self.assertFalse(spec1.is_compatible_with(spec3))
    self.assertFalse(spec1.is_compatible_with(spec4))
    self.assertFalse(spec2.is_compatible_with(spec3))
    self.assertFalse(spec2.is_compatible_with(spec4))
    self.assertFalse(spec3.is_compatible_with(spec4))
    self.assertFalse(spec4.is_compatible_with(spec5))
    value = constant_op.constant([1, 2, 3])
    self.assertFalse(spec4.is_compatible_with(value))
    self.assertTrue(spec4.is_compatible_with(WrappedTensor(value)))

  def testToList(self):
    with context.eager_mode():
      tensor_values = constant_op.constant(
          ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
      row_splits = constant_op.constant([0, 2, 2, 5, 6, 8], dtypes.int64)
      values = WrappedTensor(tensor_values)
      rt = RaggedTensor.from_row_splits(values, row_splits)
      expected = ragged_factory_ops.constant([['a', 'b'], [], ['c', 'd', 'e'],
                                              ['f'], ['g', 'h']]).to_list()

      with self.subTest('Raise on unsupported'):
        with self.assertRaisesRegex(
            ValueError,
            'values must be convertible to a list',
        ):
          _ = rt.to_list()

      with self.subTest('Value with numpy method'):

        class WrappedTensorWithNumpy(WrappedTensor):

          def numpy(self):
            return self.value.numpy()

        values = WrappedTensorWithNumpy(tensor_values)
        rt = RaggedTensor.from_row_splits(values, row_splits)
        self.assertEqual(rt.to_list(), expected)

      with self.subTest('Value with to_list method'):

        class WrappedTensorWithToList(WrappedTensor):

          def to_list(self):
            return self.value.numpy().tolist()

        values = WrappedTensorWithToList(tensor_values)
        rt = RaggedTensor.from_row_splits(values, row_splits)
        self.assertEqual(rt.to_list(), expected)

  def testFromValue(self):
    tensor_values = constant_op.constant([[1.0, 2], [4, 5], [7, 8]])
    values = WrappedTensor(tensor_values)

    row_splits = constant_op.constant([0, 2, 3, 3, 3], dtypes.int32)
    rt = RaggedTensor.from_row_splits(values, row_splits)

    rt_spec = type_spec.type_spec_from_value(rt)
    self.assertEqual(
        rt_spec,
        RaggedTensorSpec(
            shape=[4, None, 2],
            dtype=dtypes.float32,
            ragged_rank=1,
            row_splits_dtype=dtypes.int32,
            flat_values_spec=WrappedTensor.Spec(
                tensor_spec.TensorSpec([None, 2], dtypes.float32))))
    # Ensure the shape of flat_values_spec being consistent with the shape
    # of the RaggedTensor.
    self.assertEqual(rt_spec.shape[rt_spec.ragged_rank:],
                     rt_spec.flat_values_spec.shape)


if __name__ == '__main__':
  googletest.main()
