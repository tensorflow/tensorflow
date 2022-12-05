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
"""Tests for `tf.data.DatasetSpec`."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.platform import test


class DatasetSpecTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testInputSignature(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        np.arange(10).astype(np.int32)).batch(5)

    @def_function.function(input_signature=[
        dataset_ops.DatasetSpec(
            tensor_spec.TensorSpec(
                shape=(None,), dtype=dtypes.int32, name=None),
            tensor_shape.TensorShape([]))
    ])
    def fn(_):
      pass

    fn(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecInnerSpec(self):
    inner_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32)
    ds_spec = dataset_ops.DatasetSpec(inner_spec)
    self.assertEqual(ds_spec.element_spec, inner_spec)

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecTraceType(self):
    trace_type_1 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32),
        [5])
    trace_type_2 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32),
        [5])

    self.assertEqual(trace_type_1, trace_type_2)
    self.assertEqual(hash(trace_type_1), hash(trace_type_2))
    self.assertTrue(trace_type_1.is_subtype_of(trace_type_2))
    self.assertTrue(trace_type_2.is_subtype_of(trace_type_1))

    trace_type_3 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32),
        [6])
    self.assertNotEqual(trace_type_1, trace_type_3)
    self.assertFalse(trace_type_1.is_subtype_of(trace_type_3))
    self.assertFalse(trace_type_3.is_subtype_of(trace_type_1))

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecHierarchical(self):
    spec_1 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(1, None), dtype=dtypes.int32),
        [5, None, 2])
    spec_2 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(None, None), dtype=dtypes.int32),
        [None, None, None])
    spec_3 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(1, 2), dtype=dtypes.int32),
        [5, 3, 2])
    spec_4 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(None, 2), dtype=dtypes.int32),
        [None, 1, None])

    self.assertTrue(spec_1.is_subtype_of(spec_1))

    self.assertTrue(spec_1.is_subtype_of(spec_2))
    self.assertTrue(spec_3.is_subtype_of(spec_2))
    self.assertTrue(spec_4.is_subtype_of(spec_2))

    self.assertFalse(spec_2.is_subtype_of(spec_1))
    self.assertFalse(spec_2.is_subtype_of(spec_3))
    self.assertFalse(spec_2.is_subtype_of(spec_4))

    self.assertEqual(spec_1.most_specific_common_supertype([]), spec_1)
    self.assertEqual(spec_1.most_specific_common_supertype([spec_4]), spec_2)
    self.assertEqual(
        spec_1.most_specific_common_supertype([spec_3, spec_4]), spec_2)
    self.assertEqual(
        spec_1.most_specific_common_supertype([spec_2, spec_3, spec_4]), spec_2)

  # TODO(b/220385675): element_spec should always be a TypeSpec.
  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecHierarchicalDict(self):
    spec_1 = dataset_ops.DatasetSpec(
        {"a": tensor_spec.TensorSpec(shape=(1, None), dtype=dtypes.int32)},
        [])
    spec_2 = dataset_ops.DatasetSpec(
        {"a": tensor_spec.TensorSpec(shape=(None, None), dtype=dtypes.int32)},
        [])
    spec_3 = dataset_ops.DatasetSpec(
        {"b": tensor_spec.TensorSpec(shape=(1, None), dtype=dtypes.int32)},
        [])
    spec_4 = dataset_ops.DatasetSpec({"b": None}, [])

    self.assertTrue(spec_1.is_subtype_of(spec_1))
    self.assertTrue(spec_1.is_subtype_of(spec_2))
    self.assertFalse(spec_2.is_subtype_of(spec_1))

    self.assertFalse(spec_1.is_subtype_of(spec_3))
    self.assertFalse(spec_3.is_subtype_of(spec_1))
    self.assertFalse(spec_2.is_subtype_of(spec_3))
    self.assertFalse(spec_3.is_subtype_of(spec_2))

    self.assertTrue(spec_4.is_subtype_of(spec_4))
    self.assertEqual(spec_4.most_specific_common_supertype([]), spec_4)
    self.assertEqual(spec_4.most_specific_common_supertype([spec_4]), spec_4)


if __name__ == "__main__":
  test.main()
