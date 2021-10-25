# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for StructuredTensorSpec."""

from absl.testing import parameterized

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.ops.structured.structured_tensor import StructuredTensorSpec
from tensorflow.python.platform import googletest


# TypeSpecs consts for fields types.
T_3 = tensor_spec.TensorSpec([3])
T_1_2 = tensor_spec.TensorSpec([1, 2])
T_1_2_8 = tensor_spec.TensorSpec([1, 2, 8])
T_1_2_3_4 = tensor_spec.TensorSpec([1, 2, 3, 4])
T_2_3 = tensor_spec.TensorSpec([2, 3])
R_1_N = ragged_tensor.RaggedTensorSpec([1, None])
R_1_N_N = ragged_tensor.RaggedTensorSpec([1, None, None])
R_2_1_N = ragged_tensor.RaggedTensorSpec([2, 1, None])

# TensorSpecs for nrows & row_splits in the _to_components encoding.
NROWS_SPEC = tensor_spec.TensorSpec([], dtypes.int64)
PARTITION_SPEC = row_partition.RowPartitionSpec()


# pylint: disable=g-long-lambda
@test_util.run_all_in_graph_and_eager_modes
class StructuredTensorSpecTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):

  # TODO(edloper): Add a subclass of TensorFlowTestCase that overrides
  # assertAllEqual etc to work with StructuredTensors.
  def assertAllEqual(self, a, b, msg=None):
    if not (isinstance(a, structured_tensor.StructuredTensor) or
            isinstance(b, structured_tensor.StructuredTensor)):
      return super(StructuredTensorSpecTest, self).assertAllEqual(a, b, msg)
    if not (isinstance(a, structured_tensor.StructuredTensor) and
            isinstance(b, structured_tensor.StructuredTensor)):
      # TODO(edloper) Add support for this once structured_factory_ops is added.
      raise ValueError('Not supported yet')

    self.assertEqual(repr(a.shape), repr(b.shape))
    self.assertEqual(set(a.field_names()), set(b.field_names()))
    for field in a.field_names():
      self.assertAllEqual(a.field_value(field), b.field_value(field))

  def assertAllTensorsEqual(self, x, y):
    assert isinstance(x, dict) and isinstance(y, dict)
    self.assertEqual(set(x), set(y))
    for key in x:
      self.assertAllEqual(x[key], y[key])

  def testConstruction(self):
    spec1_fields = dict(a=T_1_2_3_4)
    spec1 = StructuredTensorSpec([1, 2, 3], spec1_fields)
    self.assertEqual(spec1._shape, (1, 2, 3))
    self.assertEqual(spec1._field_specs, spec1_fields)

    spec2_fields = dict(a=T_1_2, b=T_1_2_8, c=R_1_N, d=R_1_N_N, s=spec1)
    spec2 = StructuredTensorSpec([1, 2], spec2_fields)
    self.assertEqual(spec2._shape, (1, 2))
    self.assertEqual(spec2._field_specs, spec2_fields)

  @parameterized.parameters([
      (None, {}, r"StructuredTensor's shape must have known rank\."),
      ([], None, r'field_specs must be a dictionary\.'),
      ([], {1: tensor_spec.TensorSpec(None)},
       r'field_specs must be a dictionary with string keys\.'),
      ([], {'x': 0},
       r'field_specs must be a dictionary with TypeSpec values\.'),
  ])
  def testConstructionErrors(self, shape, field_specs, error):
    with self.assertRaisesRegex(TypeError, error):
      structured_tensor.StructuredTensorSpec(shape, field_specs)

  def testValueType(self):
    spec1 = StructuredTensorSpec([1, 2, 3], dict(a=T_1_2))
    self.assertEqual(spec1.value_type, StructuredTensor)

  @parameterized.parameters([
      (StructuredTensorSpec([1, 2, 3], {}),
       (tensor_shape.TensorShape([1, 2, 3]), {})),
      (StructuredTensorSpec([], {'a': T_1_2}),
       (tensor_shape.TensorShape([]), {'a': T_1_2})),
      (StructuredTensorSpec([1, 2], {'a': T_1_2, 'b': R_1_N}),
       (tensor_shape.TensorShape([1, 2]), {'a': T_1_2, 'b': R_1_N})),
      (StructuredTensorSpec([], {'a': T_1_2}),
       (tensor_shape.TensorShape([]), {'a': T_1_2})),
  ])  # pyformat: disable
  def testSerialize(self, spec, expected):
    serialization = spec._serialize()
    # Note that we can only use assertEqual because none of our cases include
    # a None dimension. A TensorShape with a None dimension is never equal
    # to another TensorShape.
    self.assertEqual(serialization, expected)

  @parameterized.parameters([
      (StructuredTensorSpec([1, 2, 3], {}),
       ({}, NROWS_SPEC, (PARTITION_SPEC, PARTITION_SPEC))),
      (StructuredTensorSpec([], {'a': T_1_2}),
       ({'a': T_1_2}, (), ())),
      (StructuredTensorSpec([1, 2], {'a': T_1_2, 'b': R_1_N}),
       ({'a': T_1_2, 'b': R_1_N}, NROWS_SPEC, (PARTITION_SPEC,))),
      (StructuredTensorSpec([], {'a': T_1_2}),
       ({'a': T_1_2}, (), ())),
  ])  # pyformat: disable
  def testComponentSpecs(self, spec, expected):
    self.assertEqual(spec._component_specs, expected)

  @parameterized.parameters([
      {
          'shape': [],
          'fields': dict(x=[[1.0, 2.0]]),
          'field_specs': dict(x=T_1_2),
      },
      {
          'shape': [2],
          'fields': dict(
              a=ragged_factory_ops.constant_value([[1.0], [2.0, 3.0]]),
              b=[[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
          'field_specs': dict(a=R_1_N, b=T_2_3),
      },
  ])  # pyformat: disable
  def testToFromComponents(self, shape, fields, field_specs):
    struct = StructuredTensor.from_fields(fields, shape)
    spec = StructuredTensorSpec(shape, field_specs)
    actual_components = spec._to_components(struct)
    self.assertLen(actual_components, 3)
    self.assertAllTensorsEqual(actual_components[0], fields)
    rt_reconstructed = spec._from_components(actual_components)
    self.assertAllEqual(struct, rt_reconstructed)

  def testToFromComponentsEmptyScalar(self):
    struct = StructuredTensor.from_fields(fields={}, shape=[])
    spec = struct._type_spec
    components = spec._to_components(struct)
    rt_reconstructed = spec._from_components(components)
    self.assertAllEqual(struct, rt_reconstructed)
    self.assertEqual(components, ({}, (), ()))

  def testToFromComponentsEmptyTensor(self):
    struct = StructuredTensor.from_fields(fields={}, shape=[1, 2, 3])
    spec = struct._type_spec
    components = spec._to_components(struct)
    rt_reconstructed = spec._from_components(components)
    self.assertAllEqual(struct, rt_reconstructed)
    self.assertLen(components, 3)
    fields, nrows, row_partitions = components
    self.assertEmpty(fields)
    self.assertAllEqual(nrows, 1)
    self.assertLen(row_partitions, 2)
    self.assertIsInstance(row_partitions[0], row_partition.RowPartition)
    self.assertIsInstance(row_partitions[1], row_partition.RowPartition)
    self.assertAllEqual(row_partitions[0].row_splits(), [0, 2])
    self.assertAllEqual(row_partitions[1].row_splits(), [0, 3, 6])

  @parameterized.parameters([
      {
          'unbatched': StructuredTensorSpec([], {}),
          'batch_size': 5,
          'batched': StructuredTensorSpec([5], {}),
      },
      {
          'unbatched': StructuredTensorSpec([1, 2], {}),
          'batch_size': 5,
          'batched': StructuredTensorSpec([5, 1, 2], {}),
      },
      {
          'unbatched': StructuredTensorSpec([], dict(a=T_3, b=R_1_N)),
          'batch_size': 2,
          'batched': StructuredTensorSpec([2], dict(a=T_2_3, b=R_2_1_N)),
      }
  ])  # pyformat: disable
  def testBatchUnbatch(self, unbatched, batch_size, batched):
    self.assertEqual(unbatched._batch(batch_size), batched)
    self.assertEqual(batched._unbatch(), unbatched)

  @parameterized.parameters([
      {
          'unbatched': lambda: [
              StructuredTensor.from_fields({'a': 1, 'b': [5, 6]}),
              StructuredTensor.from_fields({'a': 2, 'b': [7, 8]})],
          'batch_size': 2,
          'batched': lambda: StructuredTensor.from_fields(shape=[2], fields={
              'a': [1, 2],
              'b': [[5, 6], [7, 8]]}),
      },
      {
          'unbatched': lambda: [
              StructuredTensor.from_fields(shape=[3], fields={
                  'a': [1, 2, 3],
                  'b': [[5, 6], [6, 7], [7, 8]]}),
              StructuredTensor.from_fields(shape=[3], fields={
                  'a': [2, 3, 4],
                  'b': [[2, 2], [3, 3], [4, 4]]})],
          'batch_size': 2,
          'batched': lambda: StructuredTensor.from_fields(shape=[2, 3], fields={
              'a': [[1, 2, 3], [2, 3, 4]],
              'b': [[[5, 6], [6, 7], [7, 8]],
                    [[2, 2], [3, 3], [4, 4]]]}),
      },
      {
          'unbatched': lambda: [
              StructuredTensor.from_fields(shape=[], fields={
                  'a': 1,
                  'b': StructuredTensor.from_fields({'x': [5]})}),
              StructuredTensor.from_fields(shape=[], fields={
                  'a': 2,
                  'b': StructuredTensor.from_fields({'x': [6]})})],
          'batch_size': 2,
          'batched': lambda: StructuredTensor.from_fields(shape=[2], fields={
              'a': [1, 2],
              'b': StructuredTensor.from_fields(shape=[2], fields={
                  'x': [[5], [6]]})}),
      },
      {
          'unbatched': lambda: [
              StructuredTensor.from_fields(shape=[], fields={
                  'Ragged3d': ragged_factory_ops.constant_value([[1, 2], [3]]),
                  'Ragged2d': ragged_factory_ops.constant_value([1]),
              }),
              StructuredTensor.from_fields(shape=[], fields={
                  'Ragged3d': ragged_factory_ops.constant_value([[1]]),
                  'Ragged2d': ragged_factory_ops.constant_value([2, 3]),
              })],
          'batch_size': 2,
          'batched': lambda: StructuredTensor.from_fields(shape=[2], fields={
              'Ragged3d': ragged_factory_ops.constant_value(
                  [[[1, 2], [3]], [[1]]]),
              'Ragged2d': ragged_factory_ops.constant_value([[1], [2, 3]]),
          }),
          'use_only_batched_spec': True,
      },
  ])  # pyformat: disable
  def testBatchUnbatchValues(self, unbatched, batch_size, batched,
                             use_only_batched_spec=False):
    batched = batched()  # Deferred init because it creates tensors.
    unbatched = unbatched()  # Deferred init because it creates tensors.

    # Test batching.
    if use_only_batched_spec:
      unbatched_spec = type_spec.type_spec_from_value(batched)._unbatch()
    else:
      unbatched_spec = type_spec.type_spec_from_value(unbatched[0])
    unbatched_tensor_lists = [unbatched_spec._to_tensor_list(st)
                              for st in unbatched]
    batched_tensor_list = [array_ops.stack(tensors)
                           for tensors in zip(*unbatched_tensor_lists)]
    actual_batched = unbatched_spec._batch(batch_size)._from_tensor_list(
        batched_tensor_list)
    self.assertTrue(
        unbatched_spec._batch(batch_size).is_compatible_with(actual_batched))
    self.assertAllEqual(actual_batched, batched)

    # Test unbatching
    batched_spec = type_spec.type_spec_from_value(batched)
    batched_tensor_list = batched_spec._to_batched_tensor_list(batched)
    unbatched_tensor_lists = zip(
        *[array_ops.unstack(tensor) for tensor in batched_tensor_list])
    actual_unbatched = [
        batched_spec._unbatch()._from_tensor_list(tensor_list)
        for tensor_list in unbatched_tensor_lists]
    self.assertLen(actual_unbatched, len(unbatched))
    for st in actual_unbatched:
      self.assertTrue(batched_spec._unbatch().is_compatible_with(st))
    for (actual, expected) in zip(actual_unbatched, unbatched):
      self.assertAllEqual(actual, expected)


if __name__ == '__main__':
  googletest.main()
